from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import discord
from discord import app_commands
from discord.ext import commands

from utils.storage import JSONStorage
from utils.prompts import get_random_prompt, get_topic_choices


PANEL_TITLE = "House of the Dragon — Roleplay Hub"
PANEL_DESCRIPTION = (
    "Answer the call of fire and blood. Forge alliances, whisper secrets, and let dragons soar.\n\n"
    "Use the controls below to create your character, choose a house, draw RP prompts, and challenge duels.\n"
    "Be kind out of character; be bold in character. Seven bless your stories."
)

# Persistent component IDs (changing these breaks persistence across restarts)
CUSTOM_ID_CREATE = "hotd:create_profile"
CUSTOM_ID_VIEW = "hotd:view_profile"
CUSTOM_ID_PROMPT = "hotd:prompt"
CUSTOM_ID_HOUSE_SELECT = "hotd:house_select"
CUSTOM_ID_LORE_SELECT = "hotd:lore_select"
CUSTOM_ID_DUEL_USER_SELECT = "hotd:duel_user_select"

HOUSE_CHOICES: List[Tuple[str, str]] = [
    ("Targaryen", "🔥 Targaryen"),
    ("Velaryon", "🌊 Velaryon"),
    ("Hightower", "🕯️ Hightower"),
    ("Stark", "🐺 Stark"),
    ("Lannister", "🦁 Lannister"),
    ("Baratheon", "🦌 Baratheon"),
    ("Strong", "🛡️ Strong"),
    ("Cole", "⚔️ Cole"),
    ("Valaryon", "🌊 Valaryon"),  # common misspelling for fun; map to Velaryon in save step if desired
    ("None", "No Declared House"),
]

LORE_ENTRIES: Dict[str, str] = {
    "The Iron Throne": (
        "Forged from a thousand blades surrendered in conquest. It cuts those unworthy; a king should never sit easy."
    ),
    "Dragons": (
        "Creatures of fire and blood, bound to riders by will and ancient rite. A dragon remembers, and a dragon chooses."
    ),
    "Valyrian Steel": (
        "Rarer than crowns, lighter than lies. Each blade is a song from a fallen empire."
    ),
    "The Small Council": (
        "Whispers in velvet. Power is tallied in glances and sealed with quills."
    ),
    "Driftmark": (
        "Seat of House Velaryon. Ships like spears, coffers like seas. Trade is its tide and steel its breaker."
    ),
}


def build_panel_embed() -> discord.Embed:
    embed = discord.Embed(
        title=PANEL_TITLE,
        description=PANEL_DESCRIPTION,
        color=discord.Color.dark_red(),
    )
    embed.set_thumbnail(url="https://i.imgur.com/1zS8M4z.png")  # generic dragon crest
    embed.add_field(
        name="Start here",
        value=(
            "- Create or update your character\n"
            "- Choose your house sigil\n"
            "- Draw a writing prompt\n"
            "- Challenge a duel (honor-bound!)"
        ),
        inline=False,
    )
    embed.set_footer(text="Fire and blood. Keep OOC respectful.")
    return embed


def make_profile_embed(user: discord.abc.User, profile: Dict[str, str]) -> discord.Embed:
    character_name = profile.get("character_name") or "Unnamed"
    house = profile.get("house") or "None"
    dragon_name = profile.get("dragon_name") or "None"
    bio = profile.get("bio") or "No bio set yet."

    desc = (
        f"**Character**: {character_name}\n"
        f"**House**: {house}\n"
        f"**Dragon**: {dragon_name}\n\n"
        f"{bio}"
    )
    embed = discord.Embed(title=f"{user.display_name} — Profile", description=desc, color=discord.Color.red())
    if hasattr(user, "avatar") and user.avatar:
        embed.set_thumbnail(url=user.avatar.url)
    return embed


class ProfileModal(discord.ui.Modal, title="Forge Your Legend"):
    def __init__(self, storage: JSONStorage) -> None:
        super().__init__(timeout=300)
        self.storage = storage
        self.character_name = discord.ui.TextInput(
            label="Character Name",
            style=discord.TextStyle.short,
            required=True,
            max_length=64,
            placeholder="Ser/Princess/Prince ...",
        )
        self.house = discord.ui.TextInput(
            label="House",
            style=discord.TextStyle.short,
            required=False,
            max_length=32,
            placeholder="Targaryen, Stark, Velaryon, ...",
        )
        self.dragon_name = discord.ui.TextInput(
            label="Dragon Name (optional)",
            style=discord.TextStyle.short,
            required=False,
            max_length=48,
            placeholder="Caraxes, Syrax, Meleys, ...",
        )
        self.bio = discord.ui.TextInput(
            label="Short Bio",
            style=discord.TextStyle.paragraph,
            required=False,
            max_length=500,
            placeholder="A few lines of history, temperament, and goals...",
        )

        self.add_item(self.character_name)
        self.add_item(self.house)
        self.add_item(self.dragon_name)
        self.add_item(self.bio)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        user_id = interaction.user.id
        profile = {
            "character_name": self.character_name.value.strip(),
            "house": (self.house.value or "").strip() or None,
            "dragon_name": (self.dragon_name.value or "").strip() or None,
            "bio": (self.bio.value or "").strip() or None,
        }
        self.storage.upsert_profile(user_id, profile)
        embed = make_profile_embed(interaction.user, profile)
        await interaction.response.send_message(
            content="Profile saved.", embed=embed, ephemeral=True
        )


class HotDControlPanelView(discord.ui.View):
    def __init__(self, storage: JSONStorage):
        super().__init__(timeout=None)
        self.storage = storage

        async def _dispatch(interaction: discord.Interaction):
            await self.on_item_interaction(interaction)

        # Row 1: Core buttons
        btn_create = discord.ui.Button(
            label="Create / Update Character",
            style=discord.ButtonStyle.danger,
            emoji="🐉",
            custom_id=CUSTOM_ID_CREATE,
        )
        btn_create.callback = _dispatch  # type: ignore[assignment]
        self.add_item(btn_create)

        btn_view = discord.ui.Button(
            label="View My Profile",
            style=discord.ButtonStyle.success,
            emoji="🛡️",
            custom_id=CUSTOM_ID_VIEW,
        )
        btn_view.callback = _dispatch  # type: ignore[assignment]
        self.add_item(btn_view)

        # Row 2: Prompts
        btn_prompt = discord.ui.Button(
            label="Draw RP Prompt",
            style=discord.ButtonStyle.primary,
            emoji="🔥",
            custom_id=CUSTOM_ID_PROMPT,
        )
        btn_prompt.callback = _dispatch  # type: ignore[assignment]
        self.add_item(btn_prompt)

        # Row 3: House select
        select_house = discord.ui.Select(
            custom_id=CUSTOM_ID_HOUSE_SELECT,
            placeholder="Choose your House",
            min_values=1,
            max_values=1,
            options=[discord.SelectOption(label=label, value=value) for value, label in HOUSE_CHOICES],
        )
        select_house.callback = _dispatch  # type: ignore[assignment]
        self.add_item(select_house)

        # Row 4: Lore select
        select_lore = discord.ui.Select(
            custom_id=CUSTOM_ID_LORE_SELECT,
            placeholder="Lore Compendium",
            min_values=1,
            max_values=1,
            options=[discord.SelectOption(label=topic, value=topic) for topic in LORE_ENTRIES.keys()],
        )
        select_lore.callback = _dispatch  # type: ignore[assignment]
        self.add_item(select_lore)

        # Row 5: Duel user select
        select_duel = discord.ui.UserSelect(
            custom_id=CUSTOM_ID_DUEL_USER_SELECT,
            placeholder="Challenge a Duel (choose an opponent)",
            min_values=1,
            max_values=1,
        )
        select_duel.callback = _dispatch  # type: ignore[assignment]
        self.add_item(select_duel)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        # You can restrict actions here if needed
        return True

    async def on_error(self, error: Exception, item: discord.ui.Item, interaction: discord.Interaction) -> None:  # type: ignore[override]
        try:
            if interaction.response.is_done():
                await interaction.followup.send("Something went wrong. Please try again.", ephemeral=True)
            else:
                await interaction.response.send_message("Something went wrong. Please try again.", ephemeral=True)
        except Exception:
            pass

    async def on_item_interaction(self, interaction: discord.Interaction) -> None:
        # Route by custom_id/component type
        component_type = interaction.data.get("component_type") if interaction.data else None  # type: ignore[assignment]
        custom_id = interaction.data.get("custom_id") if interaction.data else None  # type: ignore[assignment]

        # Buttons
        if custom_id == CUSTOM_ID_CREATE:
            modal = ProfileModal(self.storage)
            await interaction.response.send_modal(modal)
            return
        if custom_id == CUSTOM_ID_VIEW:
            profile = self.storage.get_profile(interaction.user.id) or {}
            embed = make_profile_embed(interaction.user, profile)
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        if custom_id == CUSTOM_ID_PROMPT:
            prompt = get_random_prompt()
            await interaction.response.send_message(content=f"Here is your spark:\n\n> {prompt}\n\nShare it in a thread or the channel!", ephemeral=True)
            return

        # Selects
        if custom_id == CUSTOM_ID_HOUSE_SELECT:
            sel_values = interaction.data.get("values", []) if interaction.data else []  # type: ignore[assignment]
            chosen = sel_values[0] if sel_values else "None"
            profile = self.storage.get_profile(interaction.user.id) or {}
            profile["house"] = "Velaryon" if chosen == "Valaryon" else chosen
            self.storage.upsert_profile(interaction.user.id, profile)
            await interaction.response.send_message(
                content=f"House set to: **{profile['house']}**.", ephemeral=True
            )
            return

        if custom_id == CUSTOM_ID_LORE_SELECT:
            sel_values = interaction.data.get("values", []) if interaction.data else []  # type: ignore[assignment]
            topic = sel_values[0] if sel_values else None
            if not topic:
                await interaction.response.send_message("No lore selected.", ephemeral=True)
                return
            text = LORE_ENTRIES.get(topic, "No lore found.")
            embed = discord.Embed(title=topic, description=text, color=discord.Color.gold())
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return

        if custom_id == CUSTOM_ID_DUEL_USER_SELECT:
            users = interaction.data.get("values", []) if interaction.data else []  # type: ignore[assignment]
            target_id = int(users[0]) if users else None
            if not target_id:
                await interaction.response.send_message("Pick someone to challenge.", ephemeral=True)
                return
            if target_id == interaction.user.id:
                await interaction.response.send_message("You cannot duel yourself.", ephemeral=True)
                return
            target_user = interaction.guild.get_member(target_id) if interaction.guild else None
            if not target_user:
                await interaction.response.send_message("Could not find that member.", ephemeral=True)
                return

            await interaction.response.defer(ephemeral=True)
            duel_embed = self._resolve_and_build_duel(interaction.user, target_user)
            if interaction.channel:
                await interaction.channel.send(embed=duel_embed)
            await interaction.followup.send("Duel posted to channel.", ephemeral=True)
            return

        # Fallback
        if not interaction.response.is_done():
            await interaction.response.send_message("Unhandled interaction.", ephemeral=True)

    def _resolve_and_build_duel(self, challenger: discord.abc.User, defender: discord.abc.User) -> discord.Embed:
        challenger_profile = self.storage.get_profile(challenger.id) or {}
        defender_profile = self.storage.get_profile(defender.id) or {}

        def rating(profile: Dict[str, str]) -> float:
            base = 0.5
            if profile.get("dragon_name"):
                base += 0.05
            if (profile.get("house") or "").lower() in {"targaryen", "velaryon"}:
                base += 0.03
            return min(base, 0.9)

        p = rating(challenger_profile)
        q = rating(defender_profile)
        # Normalize to a probability challenger wins
        total = p + q
        challenger_win_prob = 0.5 if total == 0 else p / total
        challenger_wins = random.random() < challenger_win_prob

        winner = challenger if challenger_wins else defender
        loser = defender if challenger_wins else challenger

        c_name = challenger_profile.get("character_name") or challenger.display_name
        d_name = defender_profile.get("character_name") or defender.display_name

        title = f"Duel: {c_name} vs {d_name}"
        desc = (
            f"Steel sings and dragons roar in the distance.\n\n"
            f"🏆 **Victor**: {winner.mention}\n"
            f"⚔️ **Defeated**: {loser.mention}\n\n"
            f"Honor demands a rematch—or a shared cup of arbor gold."
        )

        color = discord.Color.green() if winner == challenger else discord.Color.orange()
        embed = discord.Embed(title=title, description=desc, color=color)
        embed.set_footer(text="For story and sport. Be good sports in OOC.")
        return embed

    async def on_timeout(self) -> None:
        # Never times out; required override for abstract method
        return

    async def interaction_check_item(self, interaction: discord.Interaction) -> bool:
        # not used; kept for clarity
        return True

    # No direct on_interaction hook in View; callbacks above dispatch to on_item_interaction


class HotDRPCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.storage = JSONStorage()

    def build_view(self) -> HotDControlPanelView:
        return HotDControlPanelView(self.storage)

    # ---------- Slash command group ----------
    hotd = app_commands.Group(name="hotd", description="House of the Dragon RP controls")

    @hotd.command(name="setup", description="Post the House of the Dragon RP control panel in a channel")
    @app_commands.describe(channel="Channel to post in (defaults to current)")
    @app_commands.checks.has_permissions(manage_guild=True)
    async def hotd_setup(self, interaction: discord.Interaction, channel: Optional[discord.TextChannel] = None) -> None:
        target_channel = channel or interaction.channel  # type: ignore[assignment]
        if not isinstance(target_channel, discord.TextChannel):
            await interaction.response.send_message("Please run in or choose a text channel.", ephemeral=True)
            return

        embed = build_panel_embed()
        view = self.build_view()

        try:
            message = await target_channel.send(embed=embed, view=view)
        except discord.Forbidden:
            await interaction.response.send_message("I lack permission to post there.", ephemeral=True)
            return

        self.storage.set_panel_state(interaction.guild_id, target_channel.id, message.id)  # type: ignore[arg-type]
        await interaction.response.send_message(
            f"Control panel posted in {target_channel.mention}.", ephemeral=True
        )

    @hotd.command(name="panel_refresh", description="Refresh the panel embed and controls")
    @app_commands.checks.has_permissions(manage_guild=True)
    async def hotd_panel_refresh(self, interaction: discord.Interaction) -> None:
        state = self.storage.get_panel_state(interaction.guild_id) if interaction.guild_id else None
        if not state:
            await interaction.response.send_message("No panel recorded for this server.", ephemeral=True)
            return
        channel = interaction.guild.get_channel(state["channel_id"]) if interaction.guild else None
        if not isinstance(channel, discord.TextChannel):
            await interaction.response.send_message("Saved channel not found.", ephemeral=True)
            return
        try:
            msg = await channel.fetch_message(state["message_id"])
            await msg.edit(embed=build_panel_embed(), view=self.build_view())
            await interaction.response.send_message("Panel refreshed.", ephemeral=True)
        except discord.NotFound:
            # Repost
            new_msg = await channel.send(embed=build_panel_embed(), view=self.build_view())
            self.storage.set_panel_state(interaction.guild_id, channel.id, new_msg.id)  # type: ignore[arg-type]
            await interaction.response.send_message("Old panel missing. Posted a new one.", ephemeral=True)

    @hotd.command(name="profile_view", description="View an RP profile")
    @app_commands.describe(user="Whose profile to view (defaults to you)")
    async def hotd_profile_view(self, interaction: discord.Interaction, user: Optional[discord.User] = None) -> None:
        target = user or interaction.user
        profile = self.storage.get_profile(target.id) or {}
        embed = make_profile_embed(target, profile)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @hotd.command(name="profile_set", description="Set or update your RP profile via options")
    @app_commands.describe(
        character_name="Character name",
        house="Declared house",
        dragon_name="Dragon name (optional)",
        bio="Short bio",
    )
    async def hotd_profile_set(
        self,
        interaction: discord.Interaction,
        character_name: Optional[str] = None,
        house: Optional[str] = None,
        dragon_name: Optional[str] = None,
        bio: Optional[str] = None,
    ) -> None:
        current = self.storage.get_profile(interaction.user.id) or {}
        profile = {
            "character_name": (character_name or current.get("character_name") or "Unnamed"),
            "house": (house or current.get("house")),
            "dragon_name": (dragon_name or current.get("dragon_name")),
            "bio": (bio or current.get("bio")),
        }
        self.storage.upsert_profile(interaction.user.id, profile)
        await interaction.response.send_message("Profile updated.", ephemeral=True)

    @hotd.command(name="prompt", description="Draw a public or private RP prompt")
    @app_commands.describe(topic="Optional topic filter", public="Post in channel instead of privately")
    @app_commands.choices(
        topic=[app_commands.Choice(name=t, value=t) for t in get_topic_choices()]
    )
    async def hotd_prompt(self, interaction: discord.Interaction, topic: Optional[app_commands.Choice[str]] = None, public: Optional[bool] = False) -> None:
        text = get_random_prompt(topic.value if topic else None)
        if public and interaction.channel:
            embed = discord.Embed(title="RP Prompt", description=text, color=discord.Color.blurple())
            await interaction.response.send_message("Posted to the channel.", ephemeral=True)
            await interaction.channel.send(embed=embed)
        else:
            await interaction.response.send_message(f"Here is your spark:\n\n> {text}", ephemeral=True)

    @hotd.command(name="duel", description="Challenge a duel with a member")
    @app_commands.describe(opponent="Who do you challenge?")
    async def hotd_duel(self, interaction: discord.Interaction, opponent: discord.Member) -> None:
        if opponent.id == interaction.user.id:
            await interaction.response.send_message("You cannot duel yourself.", ephemeral=True)
            return
        embed = self.build_view()._resolve_and_build_duel(interaction.user, opponent)
        if interaction.channel:
            await interaction.response.send_message("Duel posted to channel.", ephemeral=True)
            await interaction.channel.send(embed=embed)
        else:
            await interaction.response.send_message("This command must be used in a channel.", ephemeral=True)

    @hotd.command(name="profile_reset", description="Delete your RP profile")
    async def hotd_profile_reset(self, interaction: discord.Interaction) -> None:
        self.storage.delete_profile(interaction.user.id)
        await interaction.response.send_message("Your profile has been cleared.", ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    cog = HotDRPCog(bot)
    await bot.add_cog(cog)
    # Register persistent view so custom_id handlers survive restarts
    bot.add_view(cog.build_view())

