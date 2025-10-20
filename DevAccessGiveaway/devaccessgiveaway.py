from redbot.core import commands, Config  # isort:skip
from redbot.core.bot import Red  # isort:skip
from redbot.core.i18n import Translator, cog_i18n  # isort:skip
import discord  # isort:skip
import typing  # isort:skip

import asyncio
import random
from datetime import datetime, timedelta

from redbot.core.utils.chat_formatting import pagify

_: Translator = Translator("DevAccessGiveaway", __file__)


class GiveawayView(discord.ui.View):
    def __init__(self, cog, giveaway_id: str):
        super().__init__(timeout=None)
        self.cog = cog
        self.giveaway_id = giveaway_id

    @discord.ui.button(
        label="Enter Giveaway",
        style=discord.ButtonStyle.primary,
        emoji="🎉",
        custom_id="giveaway_enter"
    )
    async def enter_giveaway(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not interaction.guild:
            return

        # Check if giveaway is still active
        giveaway_data = await self.cog.config.guild(interaction.guild).giveaways.get_raw(self.giveaway_id, default=None)
        if not giveaway_data or not giveaway_data.get("active", False):
            await interaction.response.send_message("This giveaway has ended!", ephemeral=True)
            return

        # Check if user already entered
        if interaction.user.id in giveaway_data.get("participants", []):
            await interaction.response.send_message("You have already entered this giveaway!", ephemeral=True)
            return

        # Add user to participants
        participants = giveaway_data.get("participants", [])
        participants.append(interaction.user.id)
        await self.cog.config.guild(interaction.guild).giveaways.set_raw(
            self.giveaway_id, "participants", value=participants
        )

        await interaction.response.send_message("Successfully entered the giveaway! Good luck! 🎉", ephemeral=True)


@cog_i18n(_)
class DevAccessGiveaway(commands.Cog):
    """A cog for free development access giveaways!"""

    def __init__(self, bot: Red) -> None:
        self.bot = bot

        self.config: Config = Config.get_conf(
            self,
            identifier=205192943327321000143939875896557571751,  # Unique identifier
            force_registration=True,
        )
        self.CONFIG_SCHEMA: int = 1
        self.config.register_global(CONFIG_SCHEMA=None)
        self.config.register_guild(
            channel=None,
            role=None,
            giveaways={}
        )

    async def red_delete_data_for_user(self, *, requester, user_id: int) -> None:
        """Delete all user data."""
        # This cog doesn't store persistent user data
        pass

    @commands.group()
    @commands.admin_or_permissions(manage_guild=True)
    async def devgiveaway(self, ctx: commands.Context):
        """Manage development access giveaways."""
        pass

    @devgiveaway.command()
    async def setup(self, ctx: commands.Context, channel: discord.TextChannel, role: discord.Role):
        """Set up the giveaway channel and role to assign to winners."""
        await self.config.guild(ctx.guild).channel.set(channel.id)
        await self.config.guild(ctx.guild).role.set(role.id)
        
        embed = discord.Embed(
            title="✅ Giveaway Setup Complete",
            description=f"**Channel:** {channel.mention}\n**Role:** {role.mention}",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    @devgiveaway.command()
    async def create(self, ctx: commands.Context, winners: int, duration: str, *, description: str = "Free Development Access"):
        """
        Create a new giveaway.
        
        **Examples:**
        - `[p]devgiveaway create 5 1h Free development access for 5 lucky winners!`
        - `[p]devgiveaway create 3 30m Early access to new features`
        - `[p]devgiveaway create 1 1d VIP development access`
        """
        if winners < 1:
            await ctx.send("Number of winners must be at least 1!")
            return

        # Parse duration
        duration_seconds = self._parse_duration(duration)
        if duration_seconds is None:
            await ctx.send("Invalid duration format! Use formats like: 1h, 30m, 2d, 1w")
            return

        # Get configured channel and role
        channel_id = await self.config.guild(ctx.guild).channel()
        role_id = await self.config.guild(ctx.guild).role()
        
        if not channel_id or not role_id:
            await ctx.send("Please set up the giveaway channel and role first using `[p]devgiveaway setup`!")
            return

        channel = ctx.guild.get_channel(channel_id)
        role = ctx.guild.get_role(role_id)
        
        if not channel or not role:
            await ctx.send("The configured channel or role no longer exists! Please run setup again.")
            return

        # Create giveaway ID
        giveaway_id = f"giveaway_{ctx.message.id}_{int(datetime.now().timestamp())}"
        
        # Store giveaway data
        giveaway_data = {
            "active": True,
            "winners": winners,
            "description": description,
            "created_by": ctx.author.id,
            "created_at": datetime.now().isoformat(),
            "ends_at": (datetime.now() + timedelta(seconds=duration_seconds)).isoformat(),
            "participants": [],
            "message_id": None
        }
        
        await self.config.guild(ctx.guild).giveaways.set_raw(giveaway_id, value=giveaway_data)

        # Create embed
        embed = discord.Embed(
            title="🎉 Development Access Giveaway",
            description=f"**{description}**\n\n"
                       f"**Winners:** {winners}\n"
                       f"**Ends:** <t:{int((datetime.now() + timedelta(seconds=duration_seconds)).timestamp())}:R>\n\n"
                       f"Click the button below to enter!",
            color=discord.Color.blue(),
            timestamp=datetime.now() + timedelta(seconds=duration_seconds)
        )
        embed.set_footer(text=f"Giveaway ID: {giveaway_id}")

        # Create view with button
        view = GiveawayView(self, giveaway_id)
        
        # Send message
        message = await channel.send(embed=embed, view=view)
        
        # Update message ID in config
        await self.config.guild(ctx.guild).giveaways.set_raw(giveaway_id, "message_id", value=message.id)
        
        # Start the countdown task
        self.bot.loop.create_task(self._countdown_task(ctx.guild, giveaway_id, duration_seconds))
        
        await ctx.send(f"✅ Giveaway created in {channel.mention}!")

    @devgiveaway.command()
    async def list(self, ctx: commands.Context):
        """List all active giveaways."""
        giveaways = await self.config.guild(ctx.guild).giveaways()
        
        if not giveaways:
            await ctx.send("No giveaways found!")
            return

        embed = discord.Embed(
            title="🎉 Active Giveaways",
            color=discord.Color.blue()
        )
        
        for giveaway_id, data in giveaways.items():
            if data.get("active", False):
                ends_at = datetime.fromisoformat(data["ends_at"])
                embed.add_field(
                    name=f"Giveaway {giveaway_id.split('_')[1]}",
                    value=f"**Winners:** {data['winners']}\n"
                          f"**Participants:** {len(data.get('participants', []))}\n"
                          f"**Ends:** <t:{int(ends_at.timestamp())}:R>",
                    inline=False
                )
        
        await ctx.send(embed=embed)

    @devgiveaway.command()
    async def end(self, ctx: commands.Context, giveaway_id: str):
        """Manually end a giveaway."""
        giveaway_data = await self.config.guild(ctx.guild).giveaways.get_raw(giveaway_id, default=None)
        
        if not giveaway_data:
            await ctx.send("Giveaway not found!")
            return
            
        if not giveaway_data.get("active", False):
            await ctx.send("This giveaway has already ended!")
            return

        # End the giveaway
        await self._end_giveaway(ctx.guild, giveaway_id)
        await ctx.send("✅ Giveaway ended manually!")

    def _parse_duration(self, duration: str) -> typing.Optional[int]:
        """Parse duration string to seconds."""
        duration = duration.lower().strip()
        
        if duration.endswith('s'):
            return int(duration[:-1])
        elif duration.endswith('m'):
            return int(duration[:-1]) * 60
        elif duration.endswith('h'):
            return int(duration[:-1]) * 3600
        elif duration.endswith('d'):
            return int(duration[:-1]) * 86400
        elif duration.endswith('w'):
            return int(duration[:-1]) * 604800
        else:
            # Try to parse as just a number (default to minutes)
            try:
                return int(duration) * 60
            except ValueError:
                return None

    async def _countdown_task(self, guild: discord.Guild, giveaway_id: str, duration_seconds: int):
        """Background task to handle giveaway countdown."""
        await asyncio.sleep(duration_seconds)
        await self._end_giveaway(guild, giveaway_id)

    async def _end_giveaway(self, guild: discord.Guild, giveaway_id: str):
        """End a giveaway and select winners."""
        giveaway_data = await self.config.guild(guild).giveaways.get_raw(giveaway_id, default=None)
        
        if not giveaway_data or not giveaway_data.get("active", False):
            return

        participants = giveaway_data.get("participants", [])
        winners_count = giveaway_data.get("winners", 1)
        
        # Mark giveaway as inactive
        await self.config.guild(guild).giveaways.set_raw(giveaway_id, "active", value=False)
        
        # Get the message to update
        message_id = giveaway_data.get("message_id")
        channel_id = await self.config.guild(guild).channel()
        
        if message_id and channel_id:
            try:
                channel = guild.get_channel(channel_id)
                if channel:
                    message = await channel.fetch_message(message_id)
                    
                    # Update embed to show ended
                    embed = message.embeds[0]
                    embed.title = "🎉 Giveaway Ended"
                    embed.color = discord.Color.red()
                    embed.description = f"**{giveaway_data['description']}**\n\n**Winners:** {winners_count}\n**Participants:** {len(participants)}\n\n"
                    
                    if participants:
                        # Select winners
                        if len(participants) <= winners_count:
                            winners = participants
                        else:
                            winners = random.sample(participants, winners_count)
                        
                        # Assign role to winners
                        role_id = await self.config.guild(guild).role()
                        role = guild.get_role(role_id)
                        
                        if role:
                            winner_mentions = []
                            for winner_id in winners:
                                member = guild.get_member(winner_id)
                                if member:
                                    try:
                                        await member.add_roles(role, reason="Development access giveaway winner")
                                        winner_mentions.append(member.mention)
                                    except discord.Forbidden:
                                        pass
                                    except discord.HTTPException:
                                        pass
                            
                            if winner_mentions:
                                embed.description += f"**Winners:** {', '.join(winner_mentions)}"
                            else:
                                embed.description += "**Winners:** Could not assign roles (missing permissions)"
                        else:
                            embed.description += "**Winners:** Role not found (configuration error)"
                    else:
                        embed.description += "**No participants**"
                    
                    # Remove the view (button)
                    await message.edit(embed=embed, view=None)
                    
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                pass  # Message might be deleted or bot lacks permissions

    @devgiveaway.command()
    async def config(self, ctx: commands.Context):
        """Show current giveaway configuration."""
        channel_id = await self.config.guild(ctx.guild).channel()
        role_id = await self.config.guild(ctx.guild).role()
        
        embed = discord.Embed(
            title="🎉 Giveaway Configuration",
            color=discord.Color.blue()
        )
        
        if channel_id:
            channel = ctx.guild.get_channel(channel_id)
            embed.add_field(name="Channel", value=channel.mention if channel else "Channel not found", inline=False)
        else:
            embed.add_field(name="Channel", value="Not set", inline=False)
            
        if role_id:
            role = ctx.guild.get_role(role_id)
            embed.add_field(name="Role", value=role.mention if role else "Role not found", inline=False)
        else:
            embed.add_field(name="Role", value="Not set", inline=False)
        
        await ctx.send(embed=embed)
