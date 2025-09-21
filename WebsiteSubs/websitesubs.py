import asyncio
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
                "basic": None,
                "premium": None,
                "pro": None,
                "enterprise": None
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
            
        tier = sub_data.get("tier", "basic")
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
        
        # Setup early access role
        early_access_role = await self._get_or_create_role(ctx, "Early Access", "Role for website subscribers with early access")
        await self.config.guild(guild).early_access_role.set(early_access_role.id)
        
        # Setup tier roles
        tier_roles = {}
        tiers = ["basic", "premium", "pro", "enterprise"]
        
        for tier in tiers:
            role_name = f"{tier.title()} Subscriber"
            role = await self._get_or_create_role(ctx, role_name, f"Role for {tier} tier subscribers")
            tier_roles[tier] = role.id
        
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
    async def give_subscription(self, ctx, member: discord.Member, tier: str = "basic"):
        """Give subscription roles to a member.
        
        Tiers: basic, premium, pro, enterprise
        """
        tier = tier.lower()
        valid_tiers = ["basic", "premium", "pro", "enterprise"]
        
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
            "expires_at": expires_at.isoformat()
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
        """
        tier = tier.lower()
        valid_tiers = ["basic", "premium", "pro", "enterprise"]
        
        if tier not in valid_tiers:
            await ctx.send(f"❌ Invalid tier. Valid tiers: {', '.join(valid_tiers)}")
            return
        
        # Parse date
        if date_str:
            try:
                given_date = datetime.strptime(date_str, "%Y-%m-%d")
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
            "expires_at": expires_at.isoformat()
        }
        
        subscriptions = await self.config.guild(guild).subscriptions()
        subscriptions[str(member.id)] = subscription_data
        await self.config.guild(guild).subscriptions.set(subscriptions)
        
        # Send notification
        await self._send_subscription_notification(ctx, member, subscription_data, "added")
        
        embed = discord.Embed(
            title="✅ Current Subscriber Added",
            description=f"Successfully added {member.mention} as a current {tier.title()} subscriber.",
            color=discord.Color.green()
        )
        embed.add_field(name="Subscription Date", value=given_date.strftime("%Y-%m-%d"), inline=True)
        embed.add_field(name="Expires", value=f"<t:{int(expires_at.timestamp())}:R>", inline=True)
        
        await ctx.send(embed=embed)

    async def _send_subscription_notification(self, ctx, member: discord.Member, sub_data: dict, action: str):
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
        
        view = SubscriptionVerificationView(self, member, sub_data)
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
            tier_info.append(f"**{tier.title()}**: {role.mention if role else 'Not set'}")
        
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
    
    def __init__(self, cog: WebsiteSubs, member: discord.Member, sub_data: dict):
        super().__init__(timeout=None)
        self.cog = cog
        self.member = member
        self.sub_data = sub_data

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
