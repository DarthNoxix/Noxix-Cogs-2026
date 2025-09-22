import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import discord
from discord.ext import tasks
from redbot.core import Config, commands
from redbot.core.bot import Red
from redbot.core.utils.chat_formatting import humanize_timedelta

log = logging.getLogger("red.vrt.absencemanager")


class AbsenceManager(commands.Cog):
    """
    A comprehensive absence management system that automatically maintains a central embed.
    """

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890, force_registration=True)
        
        # Configuration schema
        default_guild = {
            "enabled": False,
            "channel_id": None,
            "message_id": None,
            "authorized_roles": [],
            "embed_title": "📋 Staff Absences",
            "embed_color": 0x3498db,
            "show_avatar": True,
            "auto_remove_expired": True,
            "expired_grace_period": 24,  # hours
        }
        
        self.config.register_guild(**default_guild)
        
        # Absence data schema
        default_absence = {
            "user_id": None,
            "reason": "",
            "start_date": None,
            "end_date": None,
            "is_indefinite": False,
            "added_by": None,
            "added_at": None,
        }
        
        self.config.init_custom("ABSENCE", 1)
        self.config.register_custom("ABSENCE", **default_absence)
        
        # Start the cleanup task
        self.cleanup_task.start()

    def cog_unload(self):
        """Cleanup when cog is unloaded."""
        if self.cleanup_task.is_running():
            self.cleanup_task.cancel()

    @tasks.loop(hours=1)
    async def cleanup_task(self):
        """Periodically clean up expired absences and update embeds."""
        for guild in self.bot.guilds:
            try:
                await self._cleanup_expired_absences(guild)
                await self._update_absence_embed(guild)
            except Exception as e:
                log.error(f"Error in cleanup task for guild {guild.id}: {e}")

    @cleanup_task.before_loop
    async def before_cleanup_task(self):
        """Wait for bot to be ready before starting cleanup task."""
        await self.bot.wait_until_ready()

    async def _cleanup_expired_absences(self, guild: discord.Guild):
        """Remove expired absences if auto-removal is enabled."""
        settings = await self.config.guild(guild).all()
        if not settings["auto_remove_expired"]:
            return
            
        grace_period = timedelta(hours=settings["expired_grace_period"])
        current_time = datetime.utcnow()
        
        absences = await self.config.custom("ABSENCE").all()
        to_remove = []
        
        for absence_id, absence_data in absences.items():
            if not absence_data.get("is_indefinite") and absence_data.get("end_date"):
                end_date = datetime.fromisoformat(absence_data["end_date"])
                if current_time > end_date + grace_period:
                    to_remove.append(absence_id)
        
        for absence_id in to_remove:
            await self.config.custom("ABSENCE", absence_id).clear()

    async def _get_absence_embed(self, guild: discord.Guild) -> Optional[discord.Embed]:
        """Generate the absence embed for a guild."""
        settings = await self.config.guild(guild).all()
        if not settings["enabled"]:
            return None
            
        absences = await self.config.custom("ABSENCE").all()
        current_time = datetime.utcnow()
        
        # Filter absences for this guild
        guild_absences = []
        for absence_id, absence_data in absences.items():
            if not absence_data.get("user_id"):
                continue
                
            try:
                user = guild.get_member(absence_data["user_id"])
                if not user:
                    continue
                    
                guild_absences.append((absence_data, user))
            except (ValueError, TypeError):
                continue
        
        # Sort by start date (newest first)
        guild_absences.sort(key=lambda x: x[0].get("start_date", ""), reverse=True)
        
        embed = discord.Embed(
            title=settings["embed_title"],
            color=settings["embed_color"],
            timestamp=current_time
        )
        
        if not guild_absences:
            embed.description = "🎉 **No current absences!**\nAll staff members are available."
            embed.set_footer(text="Last updated")
            return embed
        
        description_parts = []
        for i, (absence_data, user) in enumerate(guild_absences, 1):
            # Format user mention with avatar if enabled
            if settings["show_avatar"]:
                user_display = f"{user.display_avatar.url} **{user.display_name}**"
            else:
                user_display = f"**{user.display_name}**"
            
            # Format absence period
            start_date = datetime.fromisoformat(absence_data["start_date"]) if absence_data.get("start_date") else None
            end_date = datetime.fromisoformat(absence_data["end_date"]) if absence_data.get("end_date") else None
            
            if absence_data.get("is_indefinite"):
                period = "**Indefinite**"
            elif end_date:
                if current_time > end_date:
                    period = f"**Expired** (was until {end_date.strftime('%Y-%m-%d')})"
                else:
                    time_left = humanize_timedelta(timedelta=end_date - current_time)
                    period = f"Until **{end_date.strftime('%Y-%m-%d')}** ({time_left} left)"
            else:
                period = "**Unknown duration**"
            
            # Add reason if provided
            reason = absence_data.get("reason", "")
            if reason:
                reason_text = f"\n*{reason}*"
            else:
                reason_text = ""
            
            description_parts.append(f"{i}. {user_display}\n   {period}{reason_text}")
        
        embed.description = "\n\n".join(description_parts)
        embed.set_footer(text=f"Last updated • {len(guild_absences)} absence(s)")
        
        return embed

    async def _update_absence_embed(self, guild: discord.Guild):
        """Update the absence embed message."""
        settings = await self.config.guild(guild).all()
        if not settings["enabled"] or not settings["channel_id"]:
            return
            
        try:
            channel = guild.get_channel(settings["channel_id"])
            if not channel:
                return
                
            embed = await self._get_absence_embed(guild)
            if not embed:
                return
                
            if settings["message_id"]:
                try:
                    message = await channel.fetch_message(settings["message_id"])
                    view = AbsenceManagementView(self, guild)
                    await message.edit(embed=embed, view=view)
                    return
                except discord.NotFound:
                    # Message was deleted, create a new one
                    pass
            
            # Create new message
            view = AbsenceManagementView(self, guild)
            message = await channel.send(embed=embed, view=view)
            await self.config.guild(guild).message_id.set(message.id)
            
        except Exception as e:
            log.error(f"Error updating absence embed for guild {guild.id}: {e}")

    @commands.group(name="absence")
    @commands.guild_only()
    async def absence(self, ctx: commands.Context):
        """Manage staff absences with automatic embed updates."""
        pass

    @absence.command(name="setup")
    @commands.admin_or_permissions(manage_guild=True)
    async def absence_setup(self, ctx: commands.Context, channel: discord.TextChannel = None):
        """Set up the absence management system."""
        if channel is None:
            channel = ctx.channel
            
        await self.config.guild(ctx.guild).enabled.set(True)
        await self.config.guild(ctx.guild).channel_id.set(channel.id)
        
        # Create initial embed
        await self._update_absence_embed(ctx.guild)
        
        embed = discord.Embed(
            title="✅ Absence Manager Setup Complete",
            description=f"Absence management has been set up in {channel.mention}.\n\n"
                       f"**Next steps:**\n"
                       f"• Use `{ctx.prefix}absence role` to set authorized roles\n"
                       f"• Use `{ctx.prefix}absence add` to add absences\n"
                       f"• The embed will automatically update when changes are made",
            color=0x2ecc71
        )
        await ctx.send(embed=embed)

    @absence.command(name="disable")
    @commands.admin_or_permissions(manage_guild=True)
    async def absence_disable(self, ctx: commands.Context):
        """Disable the absence management system."""
        await self.config.guild(ctx.guild).enabled.set(False)
        
        embed = discord.Embed(
            title="❌ Absence Manager Disabled",
            description="The absence management system has been disabled.",
            color=0xe74c3c
        )
        await ctx.send(embed=embed)

    @absence.command(name="role")
    @commands.admin_or_permissions(manage_guild=True)
    async def absence_role(self, ctx: commands.Context, role: discord.Role = None):
        """Add or remove authorized roles for absence management."""
        if role is None:
            # Show current roles
            roles = await self.config.guild(ctx.guild).authorized_roles()
            if not roles:
                embed = discord.Embed(
                    title="📋 Authorized Roles",
                    description="No authorized roles set. All administrators can manage absences.",
                    color=0x3498db
                )
            else:
                role_mentions = [ctx.guild.get_role(role_id).mention for role_id in roles if ctx.guild.get_role(role_id)]
                embed = discord.Embed(
                    title="📋 Authorized Roles",
                    description="**Current authorized roles:**\n" + "\n".join(role_mentions),
                    color=0x3498db
                )
            await ctx.send(embed=embed)
            return
            
        # Toggle role
        roles = await self.config.guild(ctx.guild).authorized_roles()
        if role.id in roles:
            roles.remove(role.id)
            action = "removed from"
        else:
            roles.append(role.id)
            action = "added to"
            
        await self.config.guild(ctx.guild).authorized_roles.set(roles)
        
        embed = discord.Embed(
            title="✅ Role Updated",
            description=f"{role.mention} has been {action} authorized roles.",
            color=0x2ecc71
        )
        await ctx.send(embed=embed)

    @absence.command(name="add")
    async def absence_add(self, ctx: commands.Context, user: discord.Member, *, reason: str = ""):
        """Add an absence for a user."""
        if not await self._check_authorization(ctx):
            return
            
        # Create modal for absence details
        modal = AbsenceAddModal(self, user, reason)
        await ctx.send_modal(modal)

    @absence.command(name="remove")
    async def absence_remove(self, ctx: commands.Context, user: discord.Member):
        """Remove an absence for a user."""
        if not await self._check_authorization(ctx):
            return
            
        # Find and remove absence
        absences = await self.config.custom("ABSENCE").all()
        removed = False
        
        for absence_id, absence_data in absences.items():
            if absence_data.get("user_id") == user.id:
                await self.config.custom("ABSENCE", absence_id).clear()
                removed = True
                break
                
        if removed:
            await self._update_absence_embed(ctx.guild)
            embed = discord.Embed(
                title="✅ Absence Removed",
                description=f"Absence for {user.mention} has been removed.",
                color=0x2ecc71
            )
        else:
            embed = discord.Embed(
                title="❌ No Absence Found",
                description=f"No active absence found for {user.mention}.",
                color=0xe74c3c
            )
            
        await ctx.send(embed=embed)

    @absence.command(name="list")
    async def absence_list(self, ctx: commands.Context):
        """View all current absences."""
        embed = await self._get_absence_embed(ctx.guild)
        if embed:
            await ctx.send(embed=embed)
        else:
            await ctx.send("Absence management is not set up for this server.")

    @absence.command(name="config")
    @commands.admin_or_permissions(manage_guild=True)
    async def absence_config(self, ctx: commands.Context):
        """View and modify absence management configuration."""
        settings = await self.config.guild(ctx.guild).all()
        
        embed = discord.Embed(
            title="⚙️ Absence Manager Configuration",
            color=0x3498db
        )
        
        embed.add_field(
            name="Status",
            value="✅ Enabled" if settings["enabled"] else "❌ Disabled",
            inline=True
        )
        
        channel = ctx.guild.get_channel(settings["channel_id"]) if settings["channel_id"] else None
        embed.add_field(
            name="Channel",
            value=channel.mention if channel else "Not set",
            inline=True
        )
        
        roles = await self.config.guild(ctx.guild).authorized_roles()
        role_mentions = [ctx.guild.get_role(role_id).mention for role_id in roles if ctx.guild.get_role(role_id)]
        embed.add_field(
            name="Authorized Roles",
            value="\n".join(role_mentions) if role_mentions else "Administrators only",
            inline=False
        )
        
        embed.add_field(
            name="Auto-remove Expired",
            value="✅ Yes" if settings["auto_remove_expired"] else "❌ No",
            inline=True
        )
        
        embed.add_field(
            name="Grace Period",
            value=f"{settings['expired_grace_period']} hours",
            inline=True
        )
        
        view = ConfigView(self, ctx.guild)
        await ctx.send(embed=embed, view=view)

    async def _check_authorization(self, ctx: commands.Context) -> bool:
        """Check if user is authorized to manage absences."""
        if ctx.author.guild_permissions.administrator:
            return True
            
        authorized_roles = await self.config.guild(ctx.guild).authorized_roles()
        user_roles = [role.id for role in ctx.author.roles]
        
        if any(role_id in user_roles for role_id in authorized_roles):
            return True
            
        embed = discord.Embed(
            title="❌ Access Denied",
            description="You don't have permission to manage absences.",
            color=0xe74c3c
        )
        await ctx.send(embed=embed)
        return False

    async def _add_absence(self, user: discord.Member, reason: str, start_date: datetime, 
                          end_date: Optional[datetime], is_indefinite: bool, added_by: discord.Member):
        """Add a new absence record."""
        absence_id = f"{user.guild.id}_{user.id}_{int(start_date.timestamp())}"
        
        await self.config.custom("ABSENCE", absence_id).user_id.set(user.id)
        await self.config.custom("ABSENCE", absence_id).reason.set(reason)
        await self.config.custom("ABSENCE", absence_id).start_date.set(start_date.isoformat())
        await self.config.custom("ABSENCE", absence_id).end_date.set(end_date.isoformat() if end_date else None)
        await self.config.custom("ABSENCE", absence_id).is_indefinite.set(is_indefinite)
        await self.config.custom("ABSENCE", absence_id).added_by.set(added_by.id)
        await self.config.custom("ABSENCE", absence_id).added_at.set(datetime.utcnow().isoformat())
        
        # Update embed
        await self._update_absence_embed(user.guild)


class AbsenceAddModal(discord.ui.Modal):
    """Modal for adding new absences."""
    
    def __init__(self, cog: AbsenceManager, user: discord.Member, initial_reason: str = ""):
        self.cog = cog
        self.user = user
        super().__init__(title=f"Add Absence - {user.display_name}")
        
        self.reason_input = discord.ui.TextInput(
            label="Reason (optional)",
            placeholder="Enter the reason for absence...",
            default=initial_reason,
            required=False,
            max_length=1000,
            style=discord.TextStyle.paragraph
        )
        self.add_item(self.reason_input)
        
        self.duration_input = discord.ui.TextInput(
            label="Duration",
            placeholder="Examples: 'until 2024-01-15', 'for 5 days', 'indefinite'",
            required=True,
            max_length=100
        )
        self.add_item(self.duration_input)

    async def on_submit(self, interaction: discord.Interaction):
        """Handle modal submission."""
        await interaction.response.defer()
        
        try:
            reason = self.reason_input.value.strip()
            duration = self.duration_input.value.strip().lower()
            
            # Parse duration
            start_date = datetime.utcnow()
            end_date = None
            is_indefinite = False
            
            if duration == "indefinite":
                is_indefinite = True
            elif duration.startswith("until "):
                try:
                    date_str = duration[6:].strip()
                    end_date = datetime.fromisoformat(date_str)
                except ValueError:
                    await interaction.followup.send(
                        "❌ Invalid date format. Use YYYY-MM-DD format.",
                        ephemeral=True
                    )
                    return
            elif duration.startswith("for "):
                try:
                    time_str = duration[4:].strip()
                    if "day" in time_str:
                        days = int(time_str.split()[0])
                        end_date = start_date + timedelta(days=days)
                    elif "week" in time_str:
                        weeks = int(time_str.split()[0])
                        end_date = start_date + timedelta(weeks=weeks)
                    elif "month" in time_str:
                        months = int(time_str.split()[0])
                        end_date = start_date + timedelta(days=months * 30)
                    else:
                        raise ValueError("Invalid time unit")
                except (ValueError, IndexError):
                    await interaction.followup.send(
                        "❌ Invalid duration format. Use 'for X days/weeks/months'.",
                        ephemeral=True
                    )
                    return
            else:
                await interaction.followup.send(
                    "❌ Invalid duration format. Use 'until YYYY-MM-DD', 'for X days', or 'indefinite'.",
                    ephemeral=True
                )
                return
            
            # Add absence
            await self.cog._add_absence(
                self.user, reason, start_date, end_date, is_indefinite, interaction.user
            )
            
            embed = discord.Embed(
                title="✅ Absence Added",
                description=f"Absence for {self.user.mention} has been added successfully.",
                color=0x2ecc71
            )
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            log.error(f"Error adding absence: {e}")
            await interaction.followup.send(
                "❌ An error occurred while adding the absence.",
                ephemeral=True
            )


class AbsenceManagementView(discord.ui.View):
    """View with buttons for absence management."""
    
    def __init__(self, cog: AbsenceManager, guild: discord.Guild):
        self.cog = cog
        self.guild = guild
        super().__init__(timeout=None)

    @discord.ui.button(label="Add Absence", style=discord.ButtonStyle.primary, emoji="➕")
    async def add_absence_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Button to add a new absence."""
        if not await self.cog._check_authorization(interaction):
            return
            
        # Create user selection modal
        modal = UserSelectModal(self.cog, "Add Absence")
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="Remove Absence", style=discord.ButtonStyle.danger, emoji="➖")
    async def remove_absence_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Button to remove an absence."""
        if not await self.cog._check_authorization(interaction):
            return
            
        # Create user selection modal
        modal = UserSelectModal(self.cog, "Remove Absence")
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="Refresh", style=discord.ButtonStyle.secondary, emoji="🔄")
    async def refresh_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Button to refresh the embed."""
        await interaction.response.defer()
        await self.cog._update_absence_embed(self.guild)
        await interaction.followup.send("✅ Embed refreshed!", ephemeral=True)

    async def _check_authorization(self, interaction: discord.Interaction) -> bool:
        """Check if user is authorized to manage absences."""
        if interaction.user.guild_permissions.administrator:
            return True
            
        authorized_roles = await self.cog.config.guild(self.guild).authorized_roles()
        user_roles = [role.id for role in interaction.user.roles]
        
        if any(role_id in user_roles for role_id in authorized_roles):
            return True
            
        await interaction.response.send_message(
            "❌ You don't have permission to manage absences.",
            ephemeral=True
        )
        return False


class UserSelectModal(discord.ui.Modal):
    """Modal for selecting a user for absence operations."""
    
    def __init__(self, cog: AbsenceManager, action: str):
        self.cog = cog
        self.action = action
        super().__init__(title=f"{action} - Select User")
        
        self.user_input = discord.ui.TextInput(
            label="User ID or Mention",
            placeholder="Enter user ID or mention the user...",
            required=True,
            max_length=100
        )
        self.add_item(self.user_input)

    async def on_submit(self, interaction: discord.Interaction):
        """Handle modal submission."""
        await interaction.response.defer()
        
        try:
            user_input = self.user_input.value.strip()
            user = None
            
            # Try to parse as user ID
            if user_input.isdigit():
                user = interaction.guild.get_member(int(user_input))
            
            # Try to parse as mention
            if not user and user_input.startswith("<@") and user_input.endswith(">"):
                user_id = user_input[2:-1]
                if user_id.startswith("!"):
                    user_id = user_id[1:]
                if user_id.isdigit():
                    user = interaction.guild.get_member(int(user_id))
            
            if not user:
                await interaction.followup.send(
                    "❌ User not found. Please provide a valid user ID or mention.",
                    ephemeral=True
                )
                return
            
            if self.action == "Add Absence":
                modal = AbsenceAddModal(self.cog, user)
                await interaction.followup.send_modal(modal)
            elif self.action == "Remove Absence":
                # Remove absence
                absences = await self.cog.config.custom("ABSENCE").all()
                removed = False
                
                for absence_id, absence_data in absences.items():
                    if absence_data.get("user_id") == user.id:
                        await self.cog.config.custom("ABSENCE", absence_id).clear()
                        removed = True
                        break
                        
                if removed:
                    await self.cog._update_absence_embed(interaction.guild)
                    embed = discord.Embed(
                        title="✅ Absence Removed",
                        description=f"Absence for {user.mention} has been removed.",
                        color=0x2ecc71
                    )
                else:
                    embed = discord.Embed(
                        title="❌ No Absence Found",
                        description=f"No active absence found for {user.mention}.",
                        color=0xe74c3c
                    )
                    
                await interaction.followup.send(embed=embed)
                
        except Exception as e:
            log.error(f"Error in user selection: {e}")
            await interaction.followup.send(
                "❌ An error occurred while processing the request.",
                ephemeral=True
            )


class ConfigView(discord.ui.View):
    """View for configuration options."""
    
    def __init__(self, cog: AbsenceManager, guild: discord.Guild):
        self.cog = cog
        self.guild = guild
        super().__init__(timeout=300)

    @discord.ui.button(label="Toggle Auto-remove", style=discord.ButtonStyle.secondary)
    async def toggle_auto_remove(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Toggle auto-removal of expired absences."""
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message(
                "❌ You need administrator permissions to modify configuration.",
                ephemeral=True
            )
            return
            
        current = await self.cog.config.guild(self.guild).auto_remove_expired()
        await self.cog.config.guild(self.guild).auto_remove_expired.set(not current)
        
        status = "enabled" if not current else "disabled"
        await interaction.response.send_message(
            f"✅ Auto-removal of expired absences has been {status}.",
            ephemeral=True
        )

    @discord.ui.button(label="Change Title", style=discord.ButtonStyle.secondary)
    async def change_title(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Change the embed title."""
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message(
                "❌ You need administrator permissions to modify configuration.",
                ephemeral=True
            )
            return
            
        modal = TitleChangeModal(self.cog, self.guild)
        await interaction.response.send_modal(modal)


class TitleChangeModal(discord.ui.Modal):
    """Modal for changing the embed title."""
    
    def __init__(self, cog: AbsenceManager, guild: discord.Guild):
        self.cog = cog
        self.guild = guild
        super().__init__(title="Change Embed Title")
        
        self.title_input = discord.ui.TextInput(
            label="New Title",
            placeholder="Enter the new embed title...",
            required=True,
            max_length=100
        )
        self.add_item(self.title_input)

    async def on_submit(self, interaction: discord.Interaction):
        """Handle modal submission."""
        await interaction.response.defer()
        
        new_title = self.title_input.value.strip()
        await self.cog.config.guild(self.guild).embed_title.set(new_title)
        await self.cog._update_absence_embed(self.guild)
        
        embed = discord.Embed(
            title="✅ Title Updated",
            description=f"Embed title has been changed to: **{new_title}**",
            color=0x2ecc71
        )
        await interaction.followup.send(embed=embed)
