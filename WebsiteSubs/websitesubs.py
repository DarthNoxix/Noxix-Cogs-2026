import asyncio
import io
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import discord
from redbot.core import Config, commands
from redbot.core.bot import Red
from redbot.core.utils.chat_formatting import box, pagify
from redbot.core.utils.menus import DEFAULT_CONTROLS, menu
from redbot.core.utils.predicates import MessagePredicate


class WebsiteSubs(commands.Cog):
    """Website subscription management with tier-based roles and automatic expiration."""

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890, force_registration=True)
        
        default_guild = {
            "early_access_role": None,
            "tier_roles": {
                "noble": None,
                "knight": None,
                "squire": None,
                "levy": None,
                "farmer": None
            },
            "notification_channel": None,
            "subscriptions": {}
        }
        
        self.config.register_guild(**default_guild)
        
        # Start the expiration checker
        self._expiration_task = asyncio.create_task(self._check_expirations())

    def cog_unload(self):
        """Clean up when cog is unloaded."""
        if hasattr(self, '_expiration_task'):
            self._expiration_task.cancel()

    async def _check_expirations(self):
        """Check for expired subscriptions every hour."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                await self._process_expirations()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in expiration checker: {e}")

    async def _process_expirations(self):
        """Process expired subscriptions."""
        for guild_id in self.bot.guilds:
            guild = self.bot.get_guild(guild_id)
            if not guild:
                continue
                
            subscriptions = await self.config.guild(guild).subscriptions()
            expired_users = []
            
            for user_id, sub_data in subscriptions.items():
                if sub_data.get("expires_at"):
                    expires_at = datetime.fromisoformat(sub_data["expires_at"])
                    if datetime.now() >= expires_at:
                        expired_users.append((user_id, sub_data))
            
            for user_id, sub_data in expired_users:
                await self._remove_subscription_roles(guild, user_id, sub_data, expired=True)

    async def _remove_subscription_roles(self, guild: discord.Guild, user_id: int, sub_data: dict, expired: bool = False):
        """Remove subscription roles from a user."""
        member = guild.get_member(user_id)
        if not member:
            return
            
        early_access_role = guild.get_role(await self.config.guild(guild).early_access_role())
        tier_roles = await self.config.guild(guild).tier_roles()
        
        roles_to_remove = []
        
        if early_access_role and early_access_role in member.roles:
            roles_to_remove.append(early_access_role)
            
        tier = sub_data.get("tier", "farmer")
        tier_role_id = tier_roles.get(tier)
        if tier_role_id:
            tier_role = guild.get_role(tier_role_id)
            if tier_role and tier_role in member.roles:
                roles_to_remove.append(tier_role)
        
        if roles_to_remove:
            try:
                await member.remove_roles(*roles_to_remove, reason="Subscription expired" if expired else "Subscription removed")
            except discord.Forbidden:
                pass
        
        # Remove from database
        subscriptions = await self.config.guild(guild).subscriptions()
        if str(user_id) in subscriptions:
            del subscriptions[str(user_id)]
            await self.config.guild(guild).subscriptions.set(subscriptions)

    @commands.group(name="websitesubs", aliases=["ws"])
    @commands.admin_or_permissions(manage_roles=True)
    async def websitesubs(self, ctx):
        """Website subscription management commands."""
        pass

    @websitesubs.command(name="setup")
    async def setup_websitesubs(self, ctx):
        """Setup the website subscription system."""
        guild = ctx.guild
        
        # Setup early access role with specific ID
        early_access_role_id = 1239757904652013578
        early_access_role = guild.get_role(early_access_role_id)
        
        if not early_access_role:
            await ctx.send(f"❌ Early Access role with ID {early_access_role_id} was not found in this server.")
            return
        
        await self.config.guild(guild).early_access_role.set(early_access_role_id)
        
        # Setup tier roles with specific IDs
        tier_roles = {
            "noble": 1197718229838200904,
            "knight": 1197718231025188915,
            "squire": 1197718392338124903,
            "levy": 1197718366585106514,
            "farmer": 1197718231583039588
        }
        
        # Verify all roles exist
        missing_roles = []
        for tier, role_id in tier_roles.items():
            role = ctx.guild.get_role(role_id)
            if not role:
                missing_roles.append(f"{tier.title()} (ID: {role_id})")
        
        if missing_roles:
            await ctx.send(f"❌ The following roles were not found in this server:\n{chr(10).join(missing_roles)}")
            return
        
        await self.config.guild(guild).tier_roles.set(tier_roles)
        
        # Setup notification channel
        notification_channel = await self._get_or_create_channel(ctx, "subscription-notifications", "Channel for subscription notifications")
        await self.config.guild(guild).notification_channel.set(notification_channel.id)
        
        embed = discord.Embed(
            title="✅ Website Subs Setup Complete",
            description="The website subscription system has been configured with:",
            color=discord.Color.green()
        )
        embed.add_field(name="Early Access Role", value=early_access_role.mention, inline=False)
        embed.add_field(name="Tier Roles", value="\n".join([f"**{tier.title()}**: <@&{role_id}>" for tier, role_id in tier_roles.items()]), inline=False)
        embed.add_field(name="Notification Channel", value=notification_channel.mention, inline=False)
        
        await ctx.send(embed=embed)

    async def _get_or_create_role(self, ctx, name: str, reason: str) -> discord.Role:
        """Get existing role or create a new one."""
        role = discord.utils.get(ctx.guild.roles, name=name)
        if not role:
            role = await ctx.guild.create_role(name=name, reason=reason)
        return role

    async def _get_or_create_channel(self, ctx, name: str, topic: str) -> discord.TextChannel:
        """Get existing channel or create a new one."""
        channel = discord.utils.get(ctx.guild.text_channels, name=name)
        if not channel:
            overwrites = {
                ctx.guild.default_role: discord.PermissionOverwrite(read_messages=False),
                ctx.guild.me: discord.PermissionOverwrite(read_messages=True, send_messages=True)
            }
            channel = await ctx.guild.create_text_channel(name, overwrites=overwrites, topic=topic)
        return channel

    @websitesubs.command(name="give")
    async def give_subscription(self, ctx, member: discord.Member, tier: str = "farmer", *, website_username: str = None):
        """Give subscription roles to a member.
        
        Tiers: noble, knight, squire, levy, farmer
        website_username: Their username on your website (optional)
        """
        tier = tier.lower()
        valid_tiers = ["noble", "knight", "squire", "levy", "farmer"]
        
        if tier not in valid_tiers:
            await ctx.send(f"❌ Invalid tier. Valid tiers: {', '.join(valid_tiers)}")
            return
        
        guild = ctx.guild
        early_access_role = guild.get_role(await self.config.guild(guild).early_access_role())
        tier_roles = await self.config.guild(guild).tier_roles()
        tier_role = guild.get_role(tier_roles.get(tier))
        
        if not early_access_role or not tier_role:
            await ctx.send("❌ Please run `[p]websitesubs setup` first to configure roles.")
            return
        
        # Add roles
        roles_to_add = [early_access_role, tier_role]
        try:
            await member.add_roles(*roles_to_add, reason=f"Website subscription - {tier} tier")
        except discord.Forbidden:
            await ctx.send("❌ I don't have permission to add roles to this user.")
            return
        
        # Store subscription data
        expires_at = datetime.now() + timedelta(days=30)
        subscription_data = {
            "tier": tier,
            "given_by": ctx.author.id,
            "given_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat(),
            "website_username": website_username
        }
        
        subscriptions = await self.config.guild(guild).subscriptions()
        subscriptions[str(member.id)] = subscription_data
        await self.config.guild(guild).subscriptions.set(subscriptions)
        
        # Send notification
        await self._send_subscription_notification(ctx, member, subscription_data, "given")
        
        embed = discord.Embed(
            title="✅ Subscription Added",
            description=f"Successfully gave {member.mention} the {tier.title()} subscription.",
            color=discord.Color.green()
        )
        embed.add_field(name="Expires", value=f"<t:{int(expires_at.timestamp())}:R>", inline=True)
        embed.add_field(name="Given by", value=ctx.author.mention, inline=True)
        if website_username:
            embed.add_field(name="Website Username", value=website_username, inline=True)
        
        await ctx.send(embed=embed)

    @websitesubs.command(name="remove")
    async def remove_subscription(self, ctx, member: discord.Member):
        """Remove subscription roles from a member."""
        guild = ctx.guild
        subscriptions = await self.config.guild(guild).subscriptions()
        
        if str(member.id) not in subscriptions:
            await ctx.send("❌ This user doesn't have an active subscription.")
            return
        
        sub_data = subscriptions[str(member.id)]
        await self._remove_subscription_roles(guild, member.id, sub_data)
        
        embed = discord.Embed(
            title="✅ Subscription Removed",
            description=f"Successfully removed subscription from {member.mention}.",
            color=discord.Color.red()
        )
        
        await ctx.send(embed=embed)

    @websitesubs.command(name="addcurrent")
    async def add_current_subscriber(self, ctx, member: discord.Member, tier: str, *, date_str: str = None):
        """Add a current subscriber with a specific date.
        
        Date format: YYYY-MM-DD (defaults to today if not provided)
        Tiers: noble, knight, squire, levy, farmer
        You can also include their website username: [p]websitesubs addcurrent @user knight 2024-01-15 username123
        """
        tier = tier.lower()
        valid_tiers = ["noble", "knight", "squire", "levy", "farmer"]
        
        if tier not in valid_tiers:
            await ctx.send(f"❌ Invalid tier. Valid tiers: {', '.join(valid_tiers)}")
            return
        
        # Parse date and website username
        website_username = None
        if date_str:
            # Check if the last part is a username (no dashes, not a date)
            parts = date_str.split()
            if len(parts) > 1:
                # Last part might be username
                potential_username = parts[-1]
                if not any(char in potential_username for char in ['-', '/']):
                    website_username = potential_username
                    date_str = ' '.join(parts[:-1])
            
            try:
                given_date = datetime.strptime(date_str.strip(), "%Y-%m-%d")
            except ValueError:
                await ctx.send("❌ Invalid date format. Use YYYY-MM-DD")
                return
        else:
            given_date = datetime.now()
        
        guild = ctx.guild
        early_access_role = guild.get_role(await self.config.guild(guild).early_access_role())
        tier_roles = await self.config.guild(guild).tier_roles()
        tier_role = guild.get_role(tier_roles.get(tier))
        
        if not early_access_role or not tier_role:
            await ctx.send("❌ Please run `[p]websitesubs setup` first to configure roles.")
            return
        
        # Add roles
        roles_to_add = [early_access_role, tier_role]
        try:
            await member.add_roles(*roles_to_add, reason=f"Current website subscriber - {tier} tier")
        except discord.Forbidden:
            await ctx.send("❌ I don't have permission to add roles to this user.")
            return
        
        # Store subscription data
        expires_at = given_date + timedelta(days=30)
        subscription_data = {
            "tier": tier,
            "given_by": ctx.author.id,
            "given_at": given_date.isoformat(),
            "expires_at": expires_at.isoformat(),
            "website_username": website_username
        }
        
        subscriptions = await self.config.guild(guild).subscriptions()
        subscriptions[str(member.id)] = subscription_data
        await self.config.guild(guild).subscriptions.set(subscriptions)
        
        # Send notification
        await self._send_subscription_notification(ctx, member, subscription_data, "added", is_current=True)
        
        embed = discord.Embed(
            title="✅ Current Subscriber Added",
            description=f"Successfully added {member.mention} as a current {tier.title()} subscriber.",
            color=discord.Color.green()
        )
        embed.add_field(name="Subscription Date", value=given_date.strftime("%Y-%m-%d"), inline=True)
        embed.add_field(name="Expires", value=f"<t:{int(expires_at.timestamp())}:R>", inline=True)
        if website_username:
            embed.add_field(name="Website Username", value=website_username, inline=True)
        
        await ctx.send(embed=embed)

    async def _send_subscription_notification(self, ctx, member: discord.Member, sub_data: dict, action: str, is_current: bool = False):
        """Send subscription notification to the configured channel."""
        guild = ctx.guild
        notification_channel_id = await self.config.guild(guild).notification_channel()
        
        if not notification_channel_id:
            return
        
        channel = guild.get_channel(notification_channel_id)
        if not channel:
            return
        
        given_at = datetime.fromisoformat(sub_data["given_at"])
        expires_at = datetime.fromisoformat(sub_data["expires_at"])
        given_by = guild.get_member(sub_data["given_by"])
        
        embed = discord.Embed(
            title=f"📋 Subscription {action.title()}",
            color=discord.Color.blue()
        )
        embed.add_field(name="User", value=f"{member.mention} ({member.id})", inline=True)
        embed.add_field(name="Tier", value=sub_data["tier"].title(), inline=True)
        embed.add_field(name="Given by", value=given_by.mention if given_by else "Unknown", inline=True)
        embed.add_field(name="Given at", value=f"<t:{int(given_at.timestamp())}:F>", inline=True)
        embed.add_field(name="Expires", value=f"<t:{int(expires_at.timestamp())}:F>", inline=True)
        embed.add_field(name="Status", value="⏳ Pending Verification", inline=True)
        
        # Add website username if provided
        website_username = sub_data.get("website_username")
        if website_username:
            embed.add_field(name="Website Username", value=website_username, inline=True)
        
        view = SubscriptionVerificationView(self, member, sub_data, is_current=is_current)
        await channel.send(embed=embed, view=view)

    @websitesubs.command(name="list")
    async def list_subscriptions(self, ctx):
        """List all active subscriptions."""
        guild = ctx.guild
        subscriptions = await self.config.guild(guild).subscriptions()
        
        if not subscriptions:
            await ctx.send("❌ No active subscriptions found.")
            return
        
        pages = []
        current_page = []
        
        for user_id, sub_data in subscriptions.items():
            member = guild.get_member(int(user_id))
            if not member:
                continue
                
            given_at = datetime.fromisoformat(sub_data["given_at"])
            expires_at = datetime.fromisoformat(sub_data["expires_at"])
            
            entry = (
                f"**{member.display_name}** ({member.id})\n"
                f"Tier: {sub_data['tier'].title()}\n"
                f"Given: <t:{int(given_at.timestamp())}:d>\n"
                f"Expires: <t:{int(expires_at.timestamp())}:R>\n"
            )
            
            # Add website username if available
            website_username = sub_data.get("website_username")
            if website_username:
                entry += f"Website: {website_username}\n"
            
            current_page.append(entry)
            
            if len(current_page) >= 5:
                pages.append("\n".join(current_page))
                current_page = []
        
        if current_page:
            pages.append("\n".join(current_page))
        
        if not pages:
            await ctx.send("❌ No active subscriptions found.")
            return
        
        embeds = []
        for i, page in enumerate(pages, 1):
            embed = discord.Embed(
                title=f"Active Subscriptions (Page {i}/{len(pages)})",
                description=page,
                color=discord.Color.blue()
            )
            embeds.append(embed)
        
        await menu(ctx, embeds, DEFAULT_CONTROLS)

    @websitesubs.command(name="check")
    async def check_subscription(self, ctx, member: discord.Member):
        """Check a member's subscription status."""
        guild = ctx.guild
        subscriptions = await self.config.guild(guild).subscriptions()
        
        if str(member.id) not in subscriptions:
            await ctx.send(f"❌ {member.mention} doesn't have an active subscription.")
            return
        
        sub_data = subscriptions[str(member.id)]
        given_at = datetime.fromisoformat(sub_data["given_at"])
        expires_at = datetime.fromisoformat(sub_data["expires_at"])
        given_by = guild.get_member(sub_data["given_by"])
        
        embed = discord.Embed(
            title=f"📋 Subscription Status - {member.display_name}",
            color=discord.Color.blue()
        )
        embed.add_field(name="Tier", value=sub_data["tier"].title(), inline=True)
        embed.add_field(name="Given by", value=given_by.mention if given_by else "Unknown", inline=True)
        embed.add_field(name="Given at", value=f"<t:{int(given_at.timestamp())}:F>", inline=True)
        embed.add_field(name="Expires", value=f"<t:{int(expires_at.timestamp())}:F>", inline=True)
        embed.add_field(name="Time Remaining", value=f"<t:{int(expires_at.timestamp())}:R>", inline=True)
        
        # Add website username if available
        website_username = sub_data.get("website_username")
        if website_username:
            embed.add_field(name="Website Username", value=website_username, inline=True)
        
        # Check current roles
        early_access_role = guild.get_role(await self.config.guild(guild).early_access_role())
        tier_roles = await self.config.guild(guild).tier_roles()
        tier_role = guild.get_role(tier_roles.get(sub_data["tier"]))
        
        current_roles = []
        if early_access_role and early_access_role in member.roles:
            current_roles.append(early_access_role.mention)
        if tier_role and tier_role in member.roles:
            current_roles.append(tier_role.mention)
        
        embed.add_field(name="Current Roles", value="\n".join(current_roles) if current_roles else "None", inline=False)
        
        await ctx.send(embed=embed)

    @websitesubs.command(name="export")
    async def export_data(self, ctx):
        """Export all subscription data to a JSON file."""
        guild = ctx.guild
        subscriptions = await self.config.guild(guild).subscriptions()
        early_access_role = await self.config.guild(guild).early_access_role()
        tier_roles = await self.config.guild(guild).tier_roles()
        notification_channel = await self.config.guild(guild).notification_channel()
        
        # Create export data
        export_data = {
            "guild_id": guild.id,
            "guild_name": guild.name,
            "export_timestamp": datetime.now().isoformat(),
            "early_access_role": early_access_role,
            "tier_roles": tier_roles,
            "notification_channel": notification_channel,
            "subscriptions": subscriptions
        }
        
        # Convert to JSON string
        import json
        json_data = json.dumps(export_data, indent=2)
        
        # Create file
        filename = f"websitesubs_export_{guild.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Send as file
        file_obj = discord.File(
            io.StringIO(json_data),
            filename=filename,
            description="Website Subs export data"
        )
        
        embed = discord.Embed(
            title="📤 Data Export Complete",
            description=f"Exported {len(subscriptions)} subscriptions and configuration data.",
            color=discord.Color.green()
        )
        embed.add_field(name="File", value=filename, inline=True)
        embed.add_field(name="Subscriptions", value=str(len(subscriptions)), inline=True)
        embed.add_field(name="Export Time", value=f"<t:{int(datetime.now().timestamp())}:F>", inline=True)
        
        await ctx.send(embed=embed, file=file_obj)

    @websitesubs.command(name="import")
    async def import_data(self, ctx):
        """Import subscription data from a JSON file."""
        # Check if user has attachment
        if not ctx.message.attachments:
            await ctx.send("❌ Please attach a JSON file to import. Use `[p]websitesubs export` to create an export file first.")
            return
        
        attachment = ctx.message.attachments[0]
        
        # Check file extension
        if not attachment.filename.lower().endswith('.json'):
            await ctx.send("❌ Please attach a JSON file (.json extension).")
            return
        
        try:
            # Download and parse the file
            content = await attachment.read()
            import json
            import_data = json.loads(content.decode('utf-8'))
            
            # Validate the data structure
            required_keys = ['guild_id', 'early_access_role', 'tier_roles', 'subscriptions']
            for key in required_keys:
                if key not in import_data:
                    await ctx.send(f"❌ Invalid export file. Missing required field: {key}")
                    return
            
            # Confirm import
            embed = discord.Embed(
                title="⚠️ Import Confirmation",
                description="This will **OVERWRITE** all current subscription data and configuration.",
                color=discord.Color.orange()
            )
            embed.add_field(name="Source Guild", value=import_data.get('guild_name', 'Unknown'), inline=True)
            embed.add_field(name="Subscriptions", value=str(len(import_data['subscriptions'])), inline=True)
            embed.add_field(name="Export Date", value=import_data.get('export_timestamp', 'Unknown'), inline=True)
            embed.add_field(name="Early Access Role", value=f"<@&{import_data['early_access_role']}>" if import_data['early_access_role'] else "Not set", inline=True)
            embed.add_field(name="Notification Channel", value=f"<#{import_data['notification_channel']}>" if import_data['notification_channel'] else "Not set", inline=True)
            
            # Show tier roles
            tier_info = []
            for tier, role_id in import_data['tier_roles'].items():
                tier_info.append(f"**{tier.title()}**: <@&{role_id}>")
            embed.add_field(name="Tier Roles", value="\n".join(tier_info) if tier_info else "Not set", inline=False)
            
            embed.add_field(
                name="⚠️ Warning",
                value="This action **CANNOT BE UNDONE**. All current data will be lost!",
                inline=False
            )
            
            view = ImportConfirmationView(self, import_data)
            await ctx.send(embed=embed, view=view)
            
        except json.JSONDecodeError:
            await ctx.send("❌ Invalid JSON file. Please check the file format.")
        except Exception as e:
            await ctx.send(f"❌ Error reading file: {str(e)}")

    @websitesubs.command(name="setearlyaccess")
    async def set_early_access_role(self, ctx, role: discord.Role):
        """Manually set the Early Access role."""
        await self.config.guild(ctx.guild).early_access_role.set(role.id)
        
        embed = discord.Embed(
            title="✅ Early Access Role Updated",
            description=f"Successfully set Early Access role to {role.mention}",
            color=discord.Color.green()
        )
        
        await ctx.send(embed=embed)

    @websitesubs.command(name="setroles")
    async def set_role_ids(self, ctx, noble: discord.Role, knight: discord.Role, squire: discord.Role, levy: discord.Role, farmer: discord.Role):
        """Manually set the role IDs for each tier."""
        tier_roles = {
            "noble": noble.id,
            "knight": knight.id,
            "squire": squire.id,
            "levy": levy.id,
            "farmer": farmer.id
        }
        
        await self.config.guild(ctx.guild).tier_roles.set(tier_roles)
        
        embed = discord.Embed(
            title="✅ Role IDs Updated",
            description="Successfully updated tier role IDs:",
            color=discord.Color.green()
        )
        embed.add_field(name="Noble", value=noble.mention, inline=True)
        embed.add_field(name="Knight", value=knight.mention, inline=True)
        embed.add_field(name="Squire", value=squire.mention, inline=True)
        embed.add_field(name="Levy", value=levy.mention, inline=True)
        embed.add_field(name="Farmer", value=farmer.mention, inline=True)
        
        await ctx.send(embed=embed)

    @websitesubs.command(name="config")
    async def show_config(self, ctx):
        """Show current configuration."""
        guild = ctx.guild
        early_access_role = guild.get_role(await self.config.guild(guild).early_access_role())
        tier_roles = await self.config.guild(guild).tier_roles()
        notification_channel = guild.get_channel(await self.config.guild(guild).notification_channel())
        
        embed = discord.Embed(
            title="⚙️ Website Subs Configuration",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="Early Access Role",
            value=early_access_role.mention if early_access_role else "Not set",
            inline=False
        )
        
        tier_info = []
        for tier, role_id in tier_roles.items():
            role = guild.get_role(role_id)
            tier_info.append(f"**{tier.title()}**: {role.mention if role else 'Not set'} (ID: {role_id})")
        
        embed.add_field(
            name="Tier Roles",
            value="\n".join(tier_info) if tier_info else "Not set",
            inline=False
        )
        
        embed.add_field(
            name="Notification Channel",
            value=notification_channel.mention if notification_channel else "Not set",
            inline=False
        )
        
        await ctx.send(embed=embed)


class SubscriptionVerificationView(discord.ui.View):
    """View for subscription verification buttons."""
    
    def __init__(self, cog: WebsiteSubs, member: discord.Member, sub_data: dict, is_current: bool = False):
        super().__init__(timeout=None)
        self.cog = cog
        self.member = member
        self.sub_data = sub_data
        self.is_current = is_current
        
        # Remove the "Leave As Is" button if this is not a current subscriber
        if not is_current:
            # Remove the "Leave As Is" button from the view
            for item in self.children:
                if hasattr(item, 'label') and item.label == "Leave As Is":
                    self.remove_item(item)
                    break

    @discord.ui.button(label="Subscribed", style=discord.ButtonStyle.green, emoji="✅")
    async def subscribed_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Extend subscription by 30 days."""
        if not interaction.user.guild_permissions.manage_roles:
            await interaction.response.send_message("❌ You don't have permission to manage subscriptions.", ephemeral=True)
            return
        
        guild = interaction.guild
        new_expires_at = datetime.now() + timedelta(days=30)
        
        # Update subscription data
        subscriptions = await self.cog.config.guild(guild).subscriptions()
        if str(self.member.id) in subscriptions:
            subscriptions[str(self.member.id)]["expires_at"] = new_expires_at.isoformat()
            await self.cog.config.guild(guild).subscriptions.set(subscriptions)
        
        # Update embed
        embed = interaction.message.embeds[0]
        embed.set_field_at(4, name="Status", value="✅ Extended by 30 days", inline=True)
        embed.add_field(name="New Expiry", value=f"<t:{int(new_expires_at.timestamp())}:F>", inline=True)
        
        await interaction.response.edit_message(embed=embed, view=None)
        
        # Send confirmation
        await interaction.followup.send(f"✅ Extended {self.member.mention}'s subscription by 30 days.", ephemeral=True)

    @discord.ui.button(label="Unsubscribed", style=discord.ButtonStyle.red, emoji="❌")
    async def unsubscribed_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Remove subscription roles."""
        if not interaction.user.guild_permissions.manage_roles:
            await interaction.response.send_message("❌ You don't have permission to manage subscriptions.", ephemeral=True)
            return
        
        await self.cog._remove_subscription_roles(interaction.guild, self.member.id, self.sub_data)
        
        # Update embed
        embed = interaction.message.embeds[0]
        embed.set_field_at(4, name="Status", value="❌ Subscription Removed", inline=True)
        
        await interaction.response.edit_message(embed=embed, view=None)
        
        # Send confirmation
        await interaction.followup.send(f"❌ Removed subscription from {self.member.mention}.", ephemeral=True)

    @discord.ui.button(label="Leave As Is", style=discord.ButtonStyle.gray, emoji="⏸️", row=1)
    async def leave_as_is_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Leave the subscription as is (only for current subscribers)."""
        if not interaction.user.guild_permissions.manage_roles:
            await interaction.response.send_message("❌ You don't have permission to manage subscriptions.", ephemeral=True)
            return
        
        # Update embed to show it's been acknowledged
        embed = interaction.message.embeds[0]
        embed.set_field_at(4, name="Status", value="⏸️ Left As Is - Will Expire Naturally", inline=True)
        
        await interaction.response.edit_message(embed=embed, view=None)
        
        # Send confirmation
        await interaction.followup.send(f"⏸️ Left {self.member.mention}'s subscription as is. It will expire naturally on <t:{int(datetime.fromisoformat(self.sub_data['expires_at']).timestamp())}:F>.", ephemeral=True)


class ImportConfirmationView(discord.ui.View):
    """View for import confirmation buttons."""
    
    def __init__(self, cog: WebsiteSubs, import_data: dict):
        super().__init__(timeout=300)  # 5 minute timeout
        self.cog = cog
        self.import_data = import_data

    @discord.ui.button(label="Confirm Import", style=discord.ButtonStyle.green, emoji="✅")
    async def confirm_import(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Confirm and execute the import."""
        if not interaction.user.guild_permissions.manage_roles:
            await interaction.response.send_message("❌ You don't have permission to manage subscriptions.", ephemeral=True)
            return
        
        guild = interaction.guild
        
        try:
            # Import the data
            await self.cog.config.guild(guild).early_access_role.set(self.import_data['early_access_role'])
            await self.cog.config.guild(guild).tier_roles.set(self.import_data['tier_roles'])
            await self.cog.config.guild(guild).notification_channel.set(self.import_data['notification_channel'])
            await self.cog.config.guild(guild).subscriptions.set(self.import_data['subscriptions'])
            
            # Update embed
            embed = interaction.message.embeds[0]
            embed.title = "✅ Import Successful"
            embed.color = discord.Color.green()
            embed.description = f"Successfully imported {len(self.import_data['subscriptions'])} subscriptions and configuration data."
            
            await interaction.response.edit_message(embed=embed, view=None)
            
            # Send confirmation
            await interaction.followup.send(f"✅ Successfully imported data from {self.import_data.get('guild_name', 'Unknown')} guild.", ephemeral=True)
            
        except Exception as e:
            await interaction.response.send_message(f"❌ Error during import: {str(e)}", ephemeral=True)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.red, emoji="❌")
    async def cancel_import(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Cancel the import."""
        if not interaction.user.guild_permissions.manage_roles:
            await interaction.response.send_message("❌ You don't have permission to manage subscriptions.", ephemeral=True)
            return
        
        # Update embed
        embed = interaction.message.embeds[0]
        embed.title = "❌ Import Cancelled"
        embed.color = discord.Color.red()
        embed.description = "Import operation was cancelled."
        
        await interaction.response.edit_message(embed=embed, view=None)
        
        # Send confirmation
        await interaction.followup.send("❌ Import cancelled.", ephemeral=True)
