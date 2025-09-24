import asyncio
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiosqlite
import discord
from discord import app_commands
from discord.ext import commands


DATA_DIR = Path(os.getenv("HOTD_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "hotd.db"


HOUSES: List[Tuple[str, str]] = [
    ("Targaryen", "🐉"),
    ("Velaryon", "⚓"),
    ("Hightower", "🕯️"),
    ("Strong", "🛡️"),
    ("Lannister", "🦁"),
    ("Baratheon", "🦌"),
    ("Cole", "⚔️"),
    ("Arryn", "🦅"),
    ("Stark", "🐺"),
]


DAILY_COINS = 25
DAILY_COOLDOWN_SEC = 60 * 60 * 24


class DataStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not initialized")
        return self._db

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self.db_path.as_posix())
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA foreign_keys=ON;")
        await self._create_tables()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _create_tables(self) -> None:
        await self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS players (
                user_id INTEGER PRIMARY KEY,
                coins INTEGER NOT NULL DEFAULT 0,
                last_daily_claim INTEGER
            );

            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL UNIQUE,
                name TEXT NOT NULL,
                bio TEXT,
                house TEXT,
                FOREIGN KEY(user_id) REFERENCES players(user_id)
            );

            CREATE TABLE IF NOT EXISTS scenes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                thread_id INTEGER NOT NULL UNIQUE,
                title TEXT NOT NULL,
                creator_id INTEGER NOT NULL,
                is_private INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS duels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                thread_id INTEGER NOT NULL UNIQUE,
                challenger_id INTEGER NOT NULL,
                opponent_id INTEGER NOT NULL,
                title TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS guild_config (
                guild_id INTEGER PRIMARY KEY,
                scene_private INTEGER NOT NULL DEFAULT 0,
                hub_message_id INTEGER
            );
            """
        )
        await self.db.commit()

    async def ensure_player(self, user_id: int) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO players (user_id) VALUES (?)",
            (user_id,),
        )
        await self.db.commit()

    async def set_character(self, user_id: int, name: str, bio: str, house: Optional[str]) -> None:
        await self.ensure_player(user_id)
        await self.db.execute(
            "INSERT INTO characters (user_id, name, bio, house) VALUES (?, ?, ?, ?)\n"
            "ON CONFLICT(user_id) DO UPDATE SET name=excluded.name, bio=excluded.bio, house=COALESCE(excluded.house, characters.house)",
            (user_id, name, bio, house),
        )
        await self.db.commit()

    async def update_house(self, user_id: int, house: str) -> None:
        await self.ensure_player(user_id)
        await self.db.execute(
            "INSERT INTO characters (user_id, name, bio, house) VALUES (?, COALESCE((SELECT name FROM characters WHERE user_id=?), 'Unnamed'), COALESCE((SELECT bio FROM characters WHERE user_id=?), ''), ?)\n"
            "ON CONFLICT(user_id) DO UPDATE SET house=excluded.house",
            (user_id, user_id, user_id, house),
        )
        await self.db.commit()

    async def get_character(self, user_id: int) -> Optional[aiosqlite.Row]:
        async with self.db.execute("SELECT * FROM characters WHERE user_id=?", (user_id,)) as cur:
            row = await cur.fetchone()
            return row

    async def set_hub_message(self, guild_id: int, message_id: int) -> None:
        await self.db.execute(
            "INSERT INTO guild_config (guild_id, hub_message_id) VALUES (?, ?)\n"
            "ON CONFLICT(guild_id) DO UPDATE SET hub_message_id=excluded.hub_message_id",
            (guild_id, message_id),
        )
        await self.db.commit()

    async def get_guild_config(self, guild_id: int) -> aiosqlite.Row:
        await self.db.execute("INSERT OR IGNORE INTO guild_config (guild_id) VALUES (?)", (guild_id,))
        await self.db.commit()
        async with self.db.execute("SELECT * FROM guild_config WHERE guild_id=?", (guild_id,)) as cur:
            return await cur.fetchone()

    async def set_scene_privacy(self, guild_id: int, private_flag: bool) -> None:
        await self.db.execute(
            "INSERT INTO guild_config (guild_id, scene_private) VALUES (?, ?)\n"
            "ON CONFLICT(guild_id) DO UPDATE SET scene_private=excluded.scene_private",
            (guild_id, 1 if private_flag else 0),
        )
        await self.db.commit()

    async def record_scene(self, guild_id: int, thread_id: int, title: str, creator_id: int, is_private: bool) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO scenes (guild_id, thread_id, title, creator_id, is_private) VALUES (?, ?, ?, ?, ?)",
            (guild_id, thread_id, title, creator_id, 1 if is_private else 0),
        )
        await self.db.commit()

    async def record_duel(self, guild_id: int, thread_id: int, title: str, challenger_id: int, opponent_id: int) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO duels (guild_id, thread_id, title, challenger_id, opponent_id) VALUES (?, ?, ?, ?, ?)",
            (guild_id, thread_id, title, challenger_id, opponent_id),
        )
        await self.db.commit()

    async def claim_daily(self, user_id: int) -> Tuple[bool, int, int]:
        await self.ensure_player(user_id)
        now = int(time.time())
        async with self.db.execute("SELECT coins, last_daily_claim FROM players WHERE user_id=?", (user_id,)) as cur:
            row = await cur.fetchone()
        last = row["last_daily_claim"] if row else None
        if last is not None and now - int(last) < DAILY_COOLDOWN_SEC:
            remaining = DAILY_COOLDOWN_SEC - (now - int(last))
            return False, row["coins"], remaining
        new_coins = (row["coins"] if row else 0) + DAILY_COINS
        await self.db.execute("UPDATE players SET coins=?, last_daily_claim=? WHERE user_id=?", (new_coins, now, user_id))
        await self.db.commit()
        return True, new_coins, 0

    async def get_balance(self, user_id: int) -> int:
        await self.ensure_player(user_id)
        async with self.db.execute("SELECT coins FROM players WHERE user_id=?", (user_id,)) as cur:
            row = await cur.fetchone()
            return int(row["coins"]) if row else 0

    async def top_coins(self, limit: int = 10) -> List[aiosqlite.Row]:
        async with self.db.execute(
            "SELECT p.user_id, p.coins, c.name, c.house FROM players p LEFT JOIN characters c ON c.user_id=p.user_id ORDER BY p.coins DESC LIMIT ?",
            (limit,),
        ) as cur:
            return await cur.fetchall()


def build_hub_embed(guild: discord.Guild) -> discord.Embed:
    title = "House of the Dragon RP Hub"
    desc = (
        "Welcome to the dance of dragons! Forge alliances, start scenes, and challenge rivals.\n\n"
        "- Create your character and swear to a House\n"
        "- Start a Scene thread for roleplay\n"
        "- Request a Duel and let the Seven decide\n"
        "- Claim your daily coins and climb the leaderboard\n"
        "Valyrian steel tongues and courteous conduct are expected."
    )
    embed = discord.Embed(title=title, description=desc, color=discord.Color.dark_red())
    embed.set_thumbnail(url="https://static.wikia.nocookie.net/gameofthrones/images/b/b9/House_Targaryen.svg")
    embed.set_footer(text=f"{guild.name} • Fire and Blood")
    return embed


class HouseSelect(discord.ui.Select):
    def __init__(self, placeholder: str = "Choose your House"):
        options = [
            discord.SelectOption(label=f"{emoji} {name}", value=name)
            for name, emoji in HOUSES
        ]
        super().__init__(
            placeholder=placeholder,
            min_values=1,
            max_values=1,
            options=options,
            custom_id="hotd:house_select",
        )

    async def callback(self, interaction: discord.Interaction) -> None:  # type: ignore[override]
        cog: "HotDRPCog" = interaction.client.get_cog("HotDRPCog")  # type: ignore
        if not cog:
            await interaction.response.send_message("Internal error: cog missing.", ephemeral=True)
            return
        house = self.values[0]
        await cog.store.update_house(interaction.user.id, house)
        await interaction.response.send_message(
            f"You now swear to House {house}. May your banners fly high!",
            ephemeral=True,
        )


class RPHubView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        # Add the persistent select to the hub panel itself
        self.add_item(HouseSelect())

    @discord.ui.button(label="Create Character", style=discord.ButtonStyle.blurple, custom_id="hotd:create_character")
    async def create_character(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:  # type: ignore[override]
        await interaction.response.send_modal(CreateCharacterModal())

    @discord.ui.button(label="Start Scene", style=discord.ButtonStyle.green, custom_id="hotd:start_scene")
    async def start_scene(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:  # type: ignore[override]
        await interaction.response.send_modal(StartSceneModal())

    @discord.ui.button(label="Request Duel", style=discord.ButtonStyle.red, custom_id="hotd:request_duel")
    async def request_duel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:  # type: ignore[override]
        await interaction.response.send_modal(RequestDuelModal())

    @discord.ui.button(label="Claim Daily", style=discord.ButtonStyle.gray, custom_id="hotd:claim_daily")
    async def claim_daily(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:  # type: ignore[override]
        cog: "HotDRPCog" = interaction.client.get_cog("HotDRPCog")  # type: ignore
        if not cog:
            await interaction.response.send_message("Internal error: cog missing.", ephemeral=True)
            return
        ok, balance, remaining = await cog.store.claim_daily(interaction.user.id)
        if ok:
            await interaction.response.send_message(
                f"You claim {DAILY_COINS} coins. New balance: {balance}.", ephemeral=True
            )
        else:
            hours = math.ceil(remaining / 3600)
            await interaction.response.send_message(
                f"You have already claimed today. Try again in ~{hours}h.", ephemeral=True
            )

    @discord.ui.button(label="Leaderboard", style=discord.ButtonStyle.gray, custom_id="hotd:leaderboard")
    async def leaderboard(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:  # type: ignore[override]
        cog: "HotDRPCog" = interaction.client.get_cog("HotDRPCog")  # type: ignore
        if not cog:
            await interaction.response.send_message("Internal error: cog missing.", ephemeral=True)
            return
        rows = await cog.store.top_coins(limit=10)
        lines = []
        for idx, row in enumerate(rows, start=1):
            name = row["name"] or f"<@{row['user_id']}>"
            house = row["house"] or "No House"
            lines.append(f"{idx}. {name} of {house} — {row['coins']} coins")
        if not lines:
            lines = ["No one has claimed coins yet."]
        embed = discord.Embed(title="Coin Leaderboard", description="\n".join(lines), color=discord.Color.gold())
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @discord.ui.button(label="Lore Compendium", style=discord.ButtonStyle.link, url="https://asoiaf.fandom.com/wiki/House_of_the_Dragon")
    async def lore_link(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:  # type: ignore[override]
        pass


class CreateCharacterModal(discord.ui.Modal, title="Create Character"):
    name_input = discord.ui.TextInput(label="Character Name", min_length=2, max_length=50)
    bio_input = discord.ui.TextInput(label="Short Bio", style=discord.TextStyle.paragraph, required=False, max_length=500)

    async def on_submit(self, interaction: discord.Interaction) -> None:  # type: ignore[override]
        cog: "HotDRPCog" = interaction.client.get_cog("HotDRPCog")  # type: ignore
        if not cog:
            await interaction.response.send_message("Internal error: cog missing.", ephemeral=True)
            return
        await cog.store.set_character(interaction.user.id, str(self.name_input), str(self.bio_input or ""), None)
        await interaction.response.send_message(
            f"Character saved as {self.name_input}. Use the house selector to swear fealty.",
            ephemeral=True,
        )


class StartSceneModal(discord.ui.Modal, title="Start Scene"):
    title_input = discord.ui.TextInput(label="Scene Title", min_length=3, max_length=80)
    private_input = discord.ui.TextInput(label="Private? (yes/no) — leave blank for server default", required=False, max_length=5)

    async def on_submit(self, interaction: discord.Interaction) -> None:  # type: ignore[override]
        cog: "HotDRPCog" = interaction.client.get_cog("HotDRPCog")  # type: ignore
        if not cog or not interaction.guild or not isinstance(interaction.channel, discord.TextChannel):
            await interaction.response.send_message("Cannot start scene here.", ephemeral=True)
            return
        cfg = await cog.store.get_guild_config(interaction.guild.id)
        is_private_default = bool(cfg["scene_private"]) if cfg else False
        user_choice = str(self.private_input or "").strip().lower()
        is_private = is_private_default if user_choice == "" else user_choice in {"y", "yes", "true"}

        thread_type = discord.ChannelType.private_thread if is_private else discord.ChannelType.public_thread
        await interaction.response.defer(ephemeral=True, thinking=True)
        thread = await interaction.channel.create_thread(name=str(self.title_input), type=thread_type)
        await cog.store.record_scene(interaction.guild.id, thread.id, str(self.title_input), interaction.user.id, is_private)
        await thread.send(
            embed=discord.Embed(
                title=f"Scene: {self.title_input}",
                description=(
                    "A new scene begins. Keep to the tone of Westeros. \n"
                    "Invite others with Add Members (for private threads)."
                ),
                color=discord.Color.dark_gray(),
            ).set_footer(text=f"Started by {interaction.user.display_name}")
        )
        await interaction.followup.send(f"Scene thread created: {thread.mention}", ephemeral=True)


class RequestDuelModal(discord.ui.Modal, title="Request Duel"):
    title_input = discord.ui.TextInput(label="Duel Title", min_length=3, max_length=80)
    opponent_input = discord.ui.TextInput(label="Opponent @mention or ID", min_length=2, max_length=50)

    async def on_submit(self, interaction: discord.Interaction) -> None:  # type: ignore[override]
        cog: "HotDRPCog" = interaction.client.get_cog("HotDRPCog")  # type: ignore
        if not cog or not interaction.guild or not isinstance(interaction.channel, discord.TextChannel):
            await interaction.response.send_message("Cannot start duel here.", ephemeral=True)
            return
        opponent: Optional[discord.Member] = None
        # Try to resolve mention
        text = str(self.opponent_input).strip()
        if len(interaction.message.mentions) > 0:  # type: ignore[attr-defined]
            opponent = interaction.message.mentions[0]  # type: ignore[attr-defined]
        if opponent is None:
            # Try by ID
            try:
                uid = int(text.strip("<@!>"))
                opponent = interaction.guild.get_member(uid) or await interaction.guild.fetch_member(uid)
            except Exception:
                pass
        if opponent is None or opponent.bot:
            await interaction.response.send_message("Could not find that opponent.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True, thinking=True)
        thread = await interaction.channel.create_thread(name=str(self.title_input), type=discord.ChannelType.public_thread)
        await cog.store.record_duel(interaction.guild.id, thread.id, str(self.title_input), interaction.user.id, opponent.id)
        await thread.send(
            embed=discord.Embed(
                title=f"Duel: {self.title_input}",
                description=f"{interaction.user.mention} has challenged {opponent.mention}!\nRoll with /hotd roll to resolve bouts.",
                color=discord.Color.red(),
            )
        )
        await interaction.followup.send(f"Duel thread created: {thread.mention}", ephemeral=True)


def parse_dice(formula: str) -> Tuple[int, List[int]]:
    # Simple NdM+K parser
    s = formula.lower().replace(" ", "")
    total = 0
    rolls: List[int] = []
    num = ""
    sign = 1
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "+-":
            sign = 1 if ch == "+" else -1
            i += 1
        elif ch.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            value = int(s[i:j])
            if j < len(s) and s[j] == "d":
                # dice term
                j += 1
                k = j
                while k < len(s) and s[k].isdigit():
                    k += 1
                sides = int(s[j:k])
                for _ in range(value):
                    r = random.randint(1, sides)
                    rolls.append(r * sign)
                    total += r * sign
                i = k
            else:
                total += value * sign
                i = j
        elif ch == "d":
            # implies 1dX
            j = i + 1
            k = j
            while k < len(s) and s[k].isdigit():
                k += 1
            sides = int(s[j:k])
            r = random.randint(1, sides)
            rolls.append(r * sign)
            total += r * sign
            i = k
        else:
            # ignore unknown
            i += 1
    return total, rolls


class HotDRPCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.store = DataStore(DB_PATH)

    async def cog_load(self) -> None:
        await self.store.connect()
        # Register persistent hub view for all future messages with matching custom_ids
        self.bot.add_view(RPHubView())

    async def cog_unload(self) -> None:
        await self.store.close()

    hotd = app_commands.Group(name="hotd", description="House of the Dragon RP utilities")

    @hotd.command(name="post_panel", description="Post the RP hub panel in a channel")
    @app_commands.describe(channel="Channel to post the hub in (defaults to current)")
    @app_commands.checks.has_permissions(manage_guild=True)
    async def post_panel(self, interaction: discord.Interaction, channel: Optional[discord.TextChannel] = None) -> None:
        channel = channel or interaction.channel  # type: ignore[assignment]
        if not isinstance(channel, discord.TextChannel) or not interaction.guild:
            await interaction.response.send_message("Please run this in a text channel.", ephemeral=True)
            return
        embed = build_hub_embed(interaction.guild)
        view = RPHubView()
        msg = await channel.send(embed=embed, view=view)
        await self.store.set_hub_message(interaction.guild.id, msg.id)
        await interaction.response.send_message(f"Panel posted in {channel.mention}", ephemeral=True)

    @hotd.command(name="config", description="Configure RP options for this server")
    @app_commands.describe(scene_private="If true, new scenes default to private threads")
    @app_commands.checks.has_permissions(manage_guild=True)
    async def config(self, interaction: discord.Interaction, scene_private: Optional[bool] = None) -> None:
        if not interaction.guild:
            await interaction.response.send_message("Run in a server.", ephemeral=True)
            return
        updated = []
        if scene_private is not None:
            await self.store.set_scene_privacy(interaction.guild.id, scene_private)
            updated.append(f"scene_private={scene_private}")
        cfg = await self.store.get_guild_config(interaction.guild.id)
        await interaction.response.send_message(
            f"Config: scene_private={bool(cfg['scene_private'])}. " + (f"Updated {', '.join(updated)}" if updated else ""),
            ephemeral=True,
        )

    @hotd.command(name="roll", description="Roll dice, e.g. 2d6+3 or d20+1")
    @app_commands.describe(formula="Dice formula: NdM+K, e.g. 2d6+3")
    async def roll(self, interaction: discord.Interaction, formula: str) -> None:
        total, rolls = parse_dice(formula)
        pretty_rolls = ", ".join(str(r) for r in rolls) if rolls else "—"
        embed = discord.Embed(title=f"Roll: {formula}", description=f"Results: {pretty_rolls}\nTotal: **{total}**", color=discord.Color.blurple())
        await interaction.response.send_message(embed=embed)

    @hotd.command(name="character", description="View your character card")
    async def character(self, interaction: discord.Interaction, user: Optional[discord.User] = None) -> None:
        target = user or interaction.user
        row = await self.store.get_character(target.id)
        if not row:
            await interaction.response.send_message("No character yet. Use the panel to create one.", ephemeral=True)
            return
        house = row["house"] or "Unaffiliated"
        embed = discord.Embed(title=row["name"], description=row["bio"] or "", color=discord.Color.dark_red())
        embed.add_field(name="House", value=house, inline=True)
        embed.set_footer(text=f"Player: {target.display_name}")
        await interaction.response.send_message(embed=embed)

    @hotd.command(name="daily", description=f"Claim {DAILY_COINS} daily coins")
    async def daily(self, interaction: discord.Interaction) -> None:
        ok, balance, remaining = await self.store.claim_daily(interaction.user.id)
        if ok:
            await interaction.response.send_message(
                f"You claim {DAILY_COINS} coins. New balance: {balance}.", ephemeral=True
            )
        else:
            hours = math.ceil(remaining / 3600)
            await interaction.response.send_message(
                f"You have already claimed today. Try again in ~{hours}h.", ephemeral=True
            )

    @hotd.command(name="balance", description="View your coin balance")
    async def balance(self, interaction: discord.Interaction, user: Optional[discord.User] = None) -> None:
        target = user or interaction.user
        coins = await self.store.get_balance(target.id)
        await interaction.response.send_message(f"{target.display_name} has {coins} coins.")


async def setup(bot: commands.Bot) -> None:  # type: ignore[override]
    await bot.add_cog(HotDRPCog(bot))

