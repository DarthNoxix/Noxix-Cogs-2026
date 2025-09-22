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
            "embed_title": "✨ Staff Absences ✨",
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
        
        # Create beautiful embed with gradient-like colors
        embed = discord.Embed(
            title=f"✨ {settings['embed_title']} ✨",
            color=settings["embed_color"],
            timestamp=current_time
        )
        
        # Add beautiful header
        embed.set_author(
            name="📊 Staff Management System",
            icon_url=guild.icon.url if guild.icon else None
        )
        
        if not guild_absences:
            embed.description = (
                "🎉 **All Clear!** 🎉\n\n"
                "✨ All staff members are currently available and ready to assist! ✨\n"
                "🌟 No active absences at this time. 🌟"
            )
            embed.set_thumbnail(url="https://cdn.discordapp.com/emojis/✅.png")
            embed.set_footer(
                text="🔄 Auto-updated • 💎 Premium Management System",
                icon_url="https://cdn.discordapp.com/emojis/🔄.png"
            )
            return embed
        
        # Create beautiful absence entries
        for i, (absence_data, user) in enumerate(guild_absences, 1):
            # Format absence period with beautiful styling
            start_date = datetime.fromisoformat(absence_data["start_date"]) if absence_data.get("start_date") else None
            end_date = datetime.fromisoformat(absence_data["end_date"]) if absence_data.get("end_date") else None
            
            # Create status emoji and color
            if absence_data.get("is_indefinite"):
                status_emoji = "♾️"
                status_text = "**Indefinite Absence**"
                status_color = "🟡"
            elif end_date:
                if current_time > end_date:
                    end_timestamp = int(end_date.timestamp())
                    status_emoji = "⏰"
                    status_text = f"**Expired** (was until <t:{end_timestamp}:d>)"
                    status_color = "🔴"
                else:
                    end_timestamp = int(end_date.timestamp())
                    status_emoji = "⏳"
                    status_text = f"**Until <t:{end_timestamp}:R>**"
                    status_color = "🟠"
            else:
                status_emoji = "❓"
                status_text = "**Unknown Duration**"
                status_color = "⚪"
            
            # Create beautiful field
            field_name = f"{status_color} {user.display_name}"
            field_value = f"{status_emoji} {status_text}"
            
            # Add reason if provided
            reason = absence_data.get("reason", "")
            if reason:
                field_value += f"\n💭 *{reason}*"
            
            # Add start date with relative timestamp
            if start_date:
                timestamp = int(start_date.timestamp())
                field_value += f"\n📅 Started: <t:{timestamp}:R>"
            
            # Add user avatar as field icon
            embed.add_field(
                name=field_name,
                value=field_value,
                inline=False
            )
        
        # Add beautiful footer with statistics
        embed.set_footer(
            text=f"📊 {len(guild_absences)} active absence{'s' if len(guild_absences) != 1 else ''} • 🔄 Auto-updated • 💎 Premium Management",
            icon_url="https://cdn.discordapp.com/emojis/📊.png"
        )
        
        # Add thumbnail
        embed.set_thumbnail(url="https://cdn.discordapp.com/emojis/📋.png")
        
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
            title="✨ Absence Manager Setup Complete! ✨",
            description=f"🎉 **Absence management has been successfully set up!** 🎉\n\n"
                       f"📍 **Channel:** {channel.mention}\n"
                       f"🔄 **Auto-updates:** Enabled\n"
                       f"💎 **Premium features:** Active\n\n"
                       f"**🚀 Next Steps:**\n"
                       f"• Use `{ctx.prefix}absence role` to set authorized roles\n"
                       f"• Use `{ctx.prefix}absence add` to add absences\n"
                       f"• The embed will automatically update when changes are made\n\n"
                       f"🌟 **Your staff absence management system is now live!** 🌟",
            color=0x2ecc71
        )
        embed.set_thumbnail(url="https://cdn.discordapp.com/emojis/✅.png")
        embed.set_footer(text="💎 Premium Absence Management System")
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
            
        # Create a view with a button to open the modal
        view = AbsenceAddView(self, user, reason)
        embed = discord.Embed(
            title="✨ Add Absence ✨",
            description=f"🎯 **Adding absence for:** {user.mention}\n\n"
                       f"💭 **Reason:** {reason if reason else 'No reason provided'}\n\n"
                       f"📝 Click the button below to set the duration and complete the absence.",
            color=0x3498db
        )
        embed.set_thumbnail(url=user.display_avatar.url)
        embed.set_footer(text="💎 Premium Absence Management System")
        await ctx.send(embed=embed, view=view)

    @absence.command(name="quickadd")
    async def absence_quickadd(self, ctx: commands.Context, user: discord.Member, duration: str, *, reason: str = ""):
        """Quickly add an absence without using a modal."""
        if not await self._check_authorization(ctx):
            return
            
        try:
            # Parse duration
            current_time = datetime.utcnow()
            end_date = None
            is_indefinite = False
            
            duration_lower = duration.lower()
            if duration_lower == "indefinite":
                is_indefinite = True
            elif duration_lower.startswith("until "):
                try:
                    date_str = duration_lower[6:].strip()
                    end_date = datetime.fromisoformat(date_str)
                except ValueError:
                    await ctx.send("❌ Invalid date format. Use YYYY-MM-DD format.")
                    return
            elif duration_lower.startswith("for "):
                try:
                    time_str = duration_lower[4:].strip()
                    if "day" in time_str:
                        days = int(time_str.split()[0])
                        end_date = current_time + timedelta(days=days)
                    elif "week" in time_str:
                        weeks = int(time_str.split()[0])
                        end_date = current_time + timedelta(weeks=weeks)
                    elif "month" in time_str:
                        months = int(time_str.split()[0])
                        end_date = current_time + timedelta(days=months * 30)
                    else:
                        raise ValueError("Invalid time unit")
                except (ValueError, IndexError):
                    await ctx.send("❌ Invalid duration format. Use 'for X days/weeks/months'.")
                    return
            else:
                await ctx.send("❌ Invalid duration format. Use 'until YYYY-MM-DD', 'for X days', or 'indefinite'.")
                return
            
            # Add absence (start_date will be set to current time in _add_absence)
            await self._add_absence(user, reason, current_time, end_date, is_indefinite, ctx.author)
            
            embed = discord.Embed(
                title="✨ Absence Added Successfully! ✨",
                description=f"🎉 **{user.display_name}** has been marked as absent.\n\n"
                           f"📅 **Duration:** {duration}\n"
                           f"💭 **Reason:** {reason if reason else 'No reason provided'}\n\n"
                           f"🔄 The absence list has been automatically updated!",
                color=0x2ecc71
            )
            embed.set_thumbnail(url=user.display_avatar.url)
            embed.set_footer(text="💎 Premium Absence Management System")
            await ctx.send(embed=embed)
            
        except Exception as e:
            log.error(f"Error adding absence: {e}")
            await ctx.send("❌ An error occurred while adding the absence.")

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
                title="✨ Absence Removed Successfully! ✨",
                description=f"🎉 **{user.display_name}** is no longer marked as absent.\n\n"
                           f"🔄 The absence list has been automatically updated!",
                color=0x2ecc71
            )
            embed.set_thumbnail(url=user.display_avatar.url)
            embed.set_footer(text="💎 Premium Absence Management System")
        else:
            embed = discord.Embed(
                title="❌ No Absence Found",
                description=f"🔍 No active absence found for {user.mention}.",
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
            title="⚙️ Absence Manager Configuration ⚙️",
            description="🔧 **Current system settings and status** 🔧",
            color=0x3498db
        )
        
        # Status field with beautiful formatting
        status_emoji = "✅" if settings["enabled"] else "❌"
        status_text = "**Enabled**" if settings["enabled"] else "**Disabled**"
        embed.add_field(
            name=f"{status_emoji} System Status",
            value=status_text,
            inline=True
        )
        
        # Channel field
        channel = ctx.guild.get_channel(settings["channel_id"]) if settings["channel_id"] else None
        channel_text = channel.mention if channel else "**Not configured**"
        embed.add_field(
            name="📍 Absence Channel",
            value=channel_text,
            inline=True
        )
        
        # Authorized roles field
        roles = await self.config.guild(ctx.guild).authorized_roles()
        role_mentions = [ctx.guild.get_role(role_id).mention for role_id in roles if ctx.guild.get_role(role_id)]
        roles_text = "\n".join(role_mentions) if role_mentions else "**Administrators only**"
        embed.add_field(
            name="🛡️ Authorized Roles",
            value=roles_text,
            inline=False
        )
        
        # Auto-remove field
        auto_remove_emoji = "✅" if settings["auto_remove_expired"] else "❌"
        auto_remove_text = "**Enabled**" if settings["auto_remove_expired"] else "**Disabled**"
        embed.add_field(
            name=f"{auto_remove_emoji} Auto-remove Expired",
            value=auto_remove_text,
            inline=True
        )
        
        # Grace period field
        embed.add_field(
            name="⏰ Grace Period",
            value=f"**{settings['expired_grace_period']} hours**",
            inline=True
        )
        
        # Add beautiful footer
        embed.set_footer(
            text="💎 Premium Absence Management System • Use buttons below to modify settings",
            icon_url="https://cdn.discordapp.com/emojis/⚙️.png"
        )
        
        view = ConfigView(self, ctx.guild)
        await ctx.send(embed=embed, view=view)

    async def _check_authorization(self, ctx_or_interaction: Union[commands.Context, discord.Interaction]) -> bool:
        """Check if user is authorized to manage absences."""
        # Handle both Context and Interaction objects
        if isinstance(ctx_or_interaction, discord.Interaction):
            user = ctx_or_interaction.user
            guild = ctx_or_interaction.guild
            is_admin = user.guild_permissions.administrator
        else:
            user = ctx_or_interaction.author
            guild = ctx_or_interaction.guild
            is_admin = user.guild_permissions.administrator
            
        if is_admin:
            return True
            
        authorized_roles = await self.config.guild(guild).authorized_roles()
        user_roles = [role.id for role in user.roles]
        
        if any(role_id in user_roles for role_id in authorized_roles):
            return True
            
        embed = discord.Embed(
            title="❌ Access Denied",
            description="You don't have permission to manage absences.",
            color=0xe74c3c
        )
        
        if isinstance(ctx_or_interaction, discord.Interaction):
            await ctx_or_interaction.response.send_message(embed=embed, ephemeral=True)
        else:
            await ctx_or_interaction.send(embed=embed)
        return False

    async def _add_absence(self, user: discord.Member, reason: str, start_date: datetime, 
                          end_date: Optional[datetime], is_indefinite: bool, added_by: discord.Member):
        """Add a new absence record."""
        # Use current time as start date (when command was executed)
        current_time = datetime.utcnow()
        absence_id = f"{user.guild.id}_{user.id}_{int(current_time.timestamp())}"
        
        await self.config.custom("ABSENCE", absence_id).user_id.set(user.id)
        await self.config.custom("ABSENCE", absence_id).reason.set(reason)
        await self.config.custom("ABSENCE", absence_id).start_date.set(current_time.isoformat())
        await self.config.custom("ABSENCE", absence_id).end_date.set(end_date.isoformat() if end_date else None)
        await self.config.custom("ABSENCE", absence_id).is_indefinite.set(is_indefinite)
        await self.config.custom("ABSENCE", absence_id).added_by.set(added_by.id)
        await self.config.custom("ABSENCE", absence_id).added_at.set(current_time.isoformat())
        
        # Update embed
        await self._update_absence_embed(user.guild)


class AbsenceAddModal(discord.ui.Modal):
    """Beautiful modal for adding new absences."""
    
    def __init__(self, cog: AbsenceManager, user: discord.Member, initial_reason: str = ""):
        self.cog = cog
        self.user = user
        super().__init__(title=f"✨ Add Absence - {user.display_name} ✨")
        
        self.reason_input = discord.ui.TextInput(
            label="💭 Reason (Optional)",
            placeholder="Enter a detailed reason for the absence...",
            default=initial_reason,
            required=False,
            max_length=1000,
            style=discord.TextStyle.paragraph
        )
        self.add_item(self.reason_input)
        
        self.duration_input = discord.ui.TextInput(
            label="📅 Duration",
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
            current_time = datetime.utcnow()
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
                        end_date = current_time + timedelta(days=days)
                    elif "week" in time_str:
                        weeks = int(time_str.split()[0])
                        end_date = current_time + timedelta(weeks=weeks)
                    elif "month" in time_str:
                        months = int(time_str.split()[0])
                        end_date = current_time + timedelta(days=months * 30)
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
            
            # Add absence (start_date will be set to current time in _add_absence)
            await self.cog._add_absence(
                self.user, reason, current_time, end_date, is_indefinite, interaction.user
            )
            
            embed = discord.Embed(
                title="✨ Absence Added Successfully! ✨",
                description=f"🎉 **{self.user.display_name}** has been marked as absent.\n\n"
                           f"📅 **Duration:** {duration}\n"
                           f"💭 **Reason:** {reason if reason else 'No reason provided'}\n\n"
                           f"🔄 The absence list has been automatically updated!",
                color=0x2ecc71
            )
            embed.set_thumbnail(url=self.user.display_avatar.url)
            embed.set_footer(text="💎 Premium Absence Management System")
            await interaction.followup.send(embed=embed, ephemeral=True)
            
        except Exception as e:
            log.error(f"Error adding absence: {e}")
            await interaction.followup.send(
                "❌ An error occurred while adding the absence.",
                ephemeral=True
            )


class AbsenceAddView(discord.ui.View):
    """View with button to open absence add modal."""
    
    def __init__(self, cog: AbsenceManager, user: discord.Member, initial_reason: str = ""):
        self.cog = cog
        self.user = user
        self.initial_reason = initial_reason
        super().__init__(timeout=300)

    @discord.ui.button(
        label="✨ Open Absence Form", 
        style=discord.ButtonStyle.primary, 
        emoji="📝",
        custom_id="open_absence_form"
    )
    async def open_modal_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Beautiful button to open the absence add modal."""
        if not await self.cog._check_authorization(interaction):
            return
            
        modal = AbsenceAddModal(self.cog, self.user, self.initial_reason)
        await interaction.response.send_modal(modal)

    async def on_timeout(self):
        """Handle view timeout."""
        for item in self.children:
            item.disabled = True


class AbsenceManagementView(discord.ui.View):
    """View with beautiful buttons for absence management."""
    
    def __init__(self, cog: AbsenceManager, guild: discord.Guild):
        self.cog = cog
        self.guild = guild
        super().__init__(timeout=None)

    @discord.ui.button(
        label="✨ Quick Add", 
        style=discord.ButtonStyle.primary, 
        emoji="⚡",
        custom_id="absence_quick_add_button"
    )
    async def quick_add_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Beautiful button for quick absence addition."""
        if not await self.cog._check_authorization(interaction):
            return
            
        # Create quick add modal
        modal = QuickAddModal(self.cog)
        await interaction.response.send_modal(modal)

    @discord.ui.button(
        label="🗑️ Remove Absence", 
        style=discord.ButtonStyle.danger, 
        emoji="➖",
        custom_id="absence_remove_button"
    )
    async def remove_absence_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Beautiful button to remove an absence."""
        if not await self.cog._check_authorization(interaction):
            return
            
        # Create user selection modal
        modal = UserSelectModal(self.cog, "🗑️ Remove Absence")
        await interaction.response.send_modal(modal)

    @discord.ui.button(
        label="🔄 Refresh", 
        style=discord.ButtonStyle.secondary, 
        emoji="✨",
        custom_id="absence_refresh_button"
    )
    async def refresh_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Beautiful button to refresh the embed."""
        await interaction.response.defer()
        await self.cog._update_absence_embed(self.guild)
        
        # Create beautiful refresh confirmation
        embed = discord.Embed(
            title="✨ Embed Refreshed! ✨",
            description="🔄 The absence list has been updated with the latest information.",
            color=0x2ecc71
        )
        embed.set_thumbnail(url="https://cdn.discordapp.com/emojis/🔄.png")
        await interaction.followup.send(embed=embed, ephemeral=True)


class QuickAddModal(discord.ui.Modal):
    """Beautiful modal for quick absence addition."""
    
    def __init__(self, cog: AbsenceManager):
        self.cog = cog
        super().__init__(title="⚡ Quick Add Absence ⚡")
        
        self.user_input = discord.ui.TextInput(
            label="👤 User (ID, mention, or name)",
            placeholder="Enter user ID, mention, or display name...",
            required=True,
            max_length=100
        )
        self.add_item(self.user_input)
        
        self.duration_input = discord.ui.TextInput(
            label="📅 Duration",
            placeholder="Examples: 'until 2024-01-15', 'for 5 days', 'indefinite'",
            required=True,
            max_length=100
        )
        self.add_item(self.duration_input)
        
        self.reason_input = discord.ui.TextInput(
            label="💭 Reason (Optional)",
            placeholder="Enter a reason for the absence...",
            required=False,
            max_length=1000,
            style=discord.TextStyle.paragraph
        )
        self.add_item(self.reason_input)

    async def on_submit(self, interaction: discord.Interaction):
        """Handle modal submission."""
        await interaction.response.defer()
        
        try:
            # Find user
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
            
            # Try to find by display name
            if not user:
                for member in interaction.guild.members:
                    if user_input.lower() in member.display_name.lower() or user_input.lower() in member.name.lower():
                        user = member
                        break
            
            if not user:
                await interaction.followup.send(
                    "❌ User not found. Please provide a valid user ID, mention, or name.",
                    ephemeral=True
                )
                return
            
            # Parse duration
            duration = self.duration_input.value.strip().lower()
            reason = self.reason_input.value.strip()
            
            current_time = datetime.utcnow()
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
                        end_date = current_time + timedelta(days=days)
                    elif "week" in time_str:
                        weeks = int(time_str.split()[0])
                        end_date = current_time + timedelta(weeks=weeks)
                    elif "month" in time_str:
                        months = int(time_str.split()[0])
                        end_date = current_time + timedelta(days=months * 30)
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
            
            # Add absence (start_date will be set to current time in _add_absence)
            await self.cog._add_absence(
                user, reason, current_time, end_date, is_indefinite, interaction.user
            )
            
            embed = discord.Embed(
                title="✨ Absence Added Successfully! ✨",
                description=f"🎉 **{user.display_name}** has been marked as absent.\n\n"
                           f"📅 **Duration:** {duration}\n"
                           f"💭 **Reason:** {reason if reason else 'No reason provided'}\n\n"
                           f"🔄 The absence list has been automatically updated!",
                color=0x2ecc71
            )
            embed.set_thumbnail(url=user.display_avatar.url)
            embed.set_footer(text="💎 Premium Absence Management System")
            await interaction.followup.send(embed=embed, ephemeral=True)
            
        except Exception as e:
            log.error(f"Error adding absence: {e}")
            await interaction.followup.send(
                "❌ An error occurred while adding the absence.",
                ephemeral=True
            )


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
                # Create a new view with button to open modal
                view = AbsenceAddView(self.cog, user)
                embed = discord.Embed(
                    title="➕ Add Absence",
                    description=f"Click the button below to add an absence for {user.mention}.",
                    color=0x3498db
                )
                await interaction.followup.send(embed=embed, view=view, ephemeral=True)
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
                    
                await interaction.followup.send(embed=embed, ephemeral=True)
                
        except Exception as e:
            log.error(f"Error in user selection: {e}")
            await interaction.followup.send(
                "❌ An error occurred while processing the request.",
                ephemeral=True
            )


class ConfigView(discord.ui.View):
    """Beautiful view for configuration options."""
    
    def __init__(self, cog: AbsenceManager, guild: discord.Guild):
        self.cog = cog
        self.guild = guild
        super().__init__(timeout=300)

    @discord.ui.button(
        label="🔄 Toggle Auto-remove", 
        style=discord.ButtonStyle.secondary,
        emoji="⚙️",
        custom_id="toggle_auto_remove"
    )
    async def toggle_auto_remove(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Beautiful button to toggle auto-removal of expired absences."""
        if not interaction.user.guild_permissions.administrator:
            embed = discord.Embed(
                title="❌ Access Denied",
                description="🔒 You need administrator permissions to modify configuration.",
                color=0xe74c3c
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
            
        current = await self.cog.config.guild(self.guild).auto_remove_expired()
        await self.cog.config.guild(self.guild).auto_remove_expired.set(not current)
        
        status = "enabled" if not current else "disabled"
        status_emoji = "✅" if not current else "❌"
        
        embed = discord.Embed(
            title=f"{status_emoji} Auto-removal Updated",
            description=f"🔄 Auto-removal of expired absences has been **{status}**.",
            color=0x2ecc71 if not current else 0xe74c3c
        )
        embed.set_footer(text="💎 Premium Absence Management System")
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @discord.ui.button(
        label="✨ Change Title", 
        style=discord.ButtonStyle.secondary,
        emoji="📝",
        custom_id="change_title"
    )
    async def change_title(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Beautiful button to change the embed title."""
        if not interaction.user.guild_permissions.administrator:
            embed = discord.Embed(
                title="❌ Access Denied",
                description="🔒 You need administrator permissions to modify configuration.",
                color=0xe74c3c
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
            
        modal = TitleChangeModal(self.cog, self.guild)
        await interaction.response.send_modal(modal)


class TitleChangeModal(discord.ui.Modal):
    """Beautiful modal for changing the embed title."""
    
    def __init__(self, cog: AbsenceManager, guild: discord.Guild):
        self.cog = cog
        self.guild = guild
        super().__init__(title="✨ Change Embed Title ✨")
        
        self.title_input = discord.ui.TextInput(
            label="📝 New Title",
            placeholder="Enter a beautiful new embed title...",
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
            title="✨ Title Updated Successfully! ✨",
            description=f"🎉 **Embed title has been changed to:**\n\n"
                       f"📝 **{new_title}**\n\n"
                       f"🔄 The absence embed has been automatically updated!",
            color=0x2ecc71
        )
        embed.set_thumbnail(url="https://cdn.discordapp.com/emojis/📝.png")
        embed.set_footer(text="💎 Premium Absence Management System")
        await interaction.followup.send(embed=embed, ephemeral=True)
