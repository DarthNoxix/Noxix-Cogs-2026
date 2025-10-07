import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

import aiohttp
import discord
import contextlib
from redbot.core import Config, commands
from redbot.core.bot import Red
from redbot.core.utils.chat_formatting import box, pagify

log = logging.getLogger("red.noxix.knowledgebot")


class KnowledgeBot(commands.Cog):
    """
    AI-powered support ticket categorization and routing system.
    
    This cog analyzes user feedback using AI and automatically routes it to
    the appropriate department channels based on categorization.
    """

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890, force_registration=True)
        
        # Default configuration
        default_guild = {
            "enabled": False,
            "n8n_webhook_url": None,
            "use_n8n": True,
            "departments": {
                "success_story": {
                    "enabled": True,
                    "channel_id": None,
                    "webhook_url": None,
                    "name": "Customer Success"
                },
                "urgent_issue": {
                    "enabled": True,
                    "channel_id": None,
                    "webhook_url": None,
                    "name": "IT Department"
                },
                "ticket": {
                    "enabled": True,
                    "channel_id": None,
                    "webhook_url": None,
                    "name": "Helpdesk"
                }
            },
            "submission_channel": None,
            "auto_categorize": True,
            "log_channel": None
        }
        
        self.config.register_guild(**default_guild)

    async def red_delete_data_for_user(self, *, requester, user_id: int):
        """No user data to delete - this cog only stores guild configuration."""
        pass

    @commands.group(name="knowledgebot", aliases=["kb"])
    @commands.guild_only()
    @commands.admin()
    async def knowledgebot(self, ctx: commands.Context):
        """Configure the KnowledgeBot system."""
        pass

    @knowledgebot.command(name="setup")
    async def setup_knowledgebot(self, ctx: commands.Context, n8n_webhook_url: str):
        """
        Set up KnowledgeBot with your n8n webhook URL.
        
        This will enable the n8n integration for AI categorization.
        """
        await self.config.guild(ctx.guild).n8n_webhook_url.set(n8n_webhook_url)
        await self.config.guild(ctx.guild).enabled.set(True)
        
        embed = discord.Embed(
            title="KnowledgeBot Setup Complete",
            description="✅ n8n webhook URL configured\n✅ KnowledgeBot enabled\n✅ n8n integration active",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    @knowledgebot.command(name="disable")
    async def disable_knowledgebot(self, ctx: commands.Context):
        """Disable the KnowledgeBot system."""
        await self.config.guild(ctx.guild).enabled.set(False)
        await ctx.send("✅ KnowledgeBot has been disabled.")

    @knowledgebot.command(name="status")
    async def status(self, ctx: commands.Context):
        """Check the current status and configuration of KnowledgeBot."""
        conf = await self.config.guild(ctx.guild).all()
        
        embed = discord.Embed(
            title="KnowledgeBot Status",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="System Status",
            value="✅ Enabled" if conf["enabled"] else "❌ Disabled",
            inline=False
        )
        
        embed.add_field(
            name="n8n Integration",
            value="✅ Configured" if conf["n8n_webhook_url"] else "❌ Not configured",
            inline=False
        )
        
        embed.add_field(
            name="Auto-categorization",
            value="✅ Enabled" if conf["auto_categorize"] else "❌ Disabled",
            inline=False
        )
        
        # Department status
        dept_status = []
        for dept_type, dept_config in conf["departments"].items():
            status = "✅" if dept_config["enabled"] else "❌"
            channel = f"<#{dept_config['channel_id']}>" if dept_config["channel_id"] else "Not set"
            dept_status.append(f"{status} **{dept_config['name']}**: {channel}")
        
        embed.add_field(
            name="Departments",
            value="\n".join(dept_status) if dept_status else "No departments configured",
            inline=False
        )
        
        await ctx.send(embed=embed)

    @knowledgebot.group(name="department")
    async def department(self, ctx: commands.Context):
        """Manage department configurations."""
        pass

    @department.command(name="set")
    async def set_department(
        self, 
        ctx: commands.Context, 
        department_type: str, 
        channel: discord.TextChannel,
        name: Optional[str] = None
    ):
        """
        Set a department channel.
        
        Department types: success_story, urgent_issue, ticket
        """
        valid_types = ["success_story", "urgent_issue", "ticket"]
        if department_type not in valid_types:
            return await ctx.send(f"Invalid department type. Valid types: {', '.join(valid_types)}")
        
        dept_name = name or department_type.replace("_", " ").title()
        
        async with self.config.guild(ctx.guild).departments() as departments:
            departments[department_type]["channel_id"] = channel.id
            departments[department_type]["name"] = dept_name
            departments[department_type]["enabled"] = True
        
        embed = discord.Embed(
            title="Department Configured",
            description=f"✅ **{dept_name}** department set to {channel.mention}",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    @department.command(name="disable")
    async def disable_department(self, ctx: commands.Context, department_type: str):
        """Disable a department."""
        valid_types = ["success_story", "urgent_issue", "ticket"]
        if department_type not in valid_types:
            return await ctx.send(f"Invalid department type. Valid types: {', '.join(valid_types)}")
        
        async with self.config.guild(ctx.guild).departments() as departments:
            departments[department_type]["enabled"] = False
        
        await ctx.send(f"✅ {department_type.replace('_', ' ').title()} department disabled.")

    @knowledgebot.command(name="submission")
    async def set_submission_channel(self, ctx: commands.Context, channel: discord.TextChannel):
        """Set the channel where users can submit feedback."""
        await self.config.guild(ctx.guild).submission_channel.set(channel.id)
        
        embed = discord.Embed(
            title="Submission Channel Set",
            description=f"✅ Users can now submit feedback in {channel.mention}",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    @knowledgebot.command(name="log")
    async def set_log_channel(self, ctx: commands.Context, channel: discord.TextChannel):
        """Set the channel for logging categorization results."""
        await self.config.guild(ctx.guild).log_channel.set(channel.id)
        
        embed = discord.Embed(
            title="Log Channel Set",
            description=f"✅ Categorization logs will be sent to {channel.mention}",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    @knowledgebot.command(name="webhook")
    async def set_webhook_url(self, ctx: commands.Context, webhook_url: str):
        """Set or update the n8n webhook URL."""
        await self.config.guild(ctx.guild).n8n_webhook_url.set(webhook_url)
        
        embed = discord.Embed(
            title="n8n Webhook URL Updated",
            description="✅ n8n webhook URL has been configured",
            color=discord.Color.green()
        )
        embed.add_field(
            name="Webhook URL",
            value=f"`{webhook_url[:50]}...`" if len(webhook_url) > 50 else f"`{webhook_url}`",
            inline=False
        )
        await ctx.send(embed=embed)

    @commands.command(name="agent")
    @commands.guild_only()
    async def submit_feedback(self, ctx: commands.Context, *, feedback: str):
        """
        Submit feedback to be categorized and routed to the appropriate department.
        
        The AI will analyze your feedback and route it to:
        - Customer Success (for success stories)
        - IT Department (for urgent issues)
        - Helpdesk (for general tickets)
        """
        conf = await self.config.guild(ctx.guild).all()
        
        if not conf["enabled"]:
            return await ctx.send("❌ KnowledgeBot is not enabled on this server.")
        
        if not conf["n8n_webhook_url"]:
            return await ctx.send("❌ KnowledgeBot is not properly configured. Please contact an administrator.")
        
        # Check if we're in the submission channel (if configured)
        if conf["submission_channel"] and ctx.channel.id != conf["submission_channel"]:
            submission_channel = self.bot.get_channel(conf["submission_channel"])
            if submission_channel:
                return await ctx.send(f"Please submit feedback in {submission_channel.mention}")
        
        try:
            # Send feedback to n8n for processing (fire-and-forget)
            success = await self.send_to_n8n(ctx, feedback, conf["n8n_webhook_url"])

            if not success:
                # React with ❌ on failure
                with contextlib.suppress(discord.Forbidden, discord.HTTPException):
                    await ctx.message.add_reaction("❌")
                return

            # React with ✅ to indicate it was sent to n8n
            try:
                await ctx.message.add_reaction("✅")
            except (discord.Forbidden, discord.HTTPException):
                pass

            # Remove the bot's reaction after 30 seconds
            try:
                await asyncio.sleep(30)
                await ctx.message.remove_reaction("✅", ctx.me)
            except (discord.Forbidden, discord.HTTPException):
                pass
            
        except Exception as e:
            log.error(f"Error processing feedback: {e}", exc_info=True)
            with contextlib.suppress(discord.Forbidden, discord.HTTPException):
                await ctx.message.add_reaction("❌")

    async def send_to_n8n(self, ctx: commands.Context, feedback: str, webhook_url: str) -> bool:
        """
        Send feedback to n8n webhook for processing.
        
        Returns True if webhook accepted (HTTP 200), otherwise False.
        """
        try:
            # Payload aligned to your Set node (Map Discord -> Chat format)
            # It expects: content, channelId, authorId
            payload = {
                "content": feedback,
                "channelId": ctx.channel.id,
                "authorId": ctx.author.id,
                # Recommended session id for n8n Simple Memory
                # per-user-in-channel conversation key
                "sessionId": f"{ctx.channel.id}:{ctx.author.id}",
                # Extra context (available if you want in the workflow)
                "user_id": ctx.author.id,
                "username": ctx.author.name,
                "discriminator": ctx.author.discriminator,
                "guild_id": ctx.guild.id,
                "guild_name": ctx.guild.name,
                "channel_name": ctx.channel.name,
                "timestamp": datetime.utcnow().isoformat(),
                "message_id": ctx.message.id
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        log.info(f"Sent to n8n for user {ctx.author.id}")
                        return True
                    log.error(f"n8n webhook returned status {response.status}")
                    return False
                        
        except asyncio.TimeoutError:
            log.error("Timeout sending feedback to n8n webhook")
            return False
        except Exception as e:
            log.error(f"Error sending feedback to n8n: {e}", exc_info=True)
            return False


    @knowledgebot.command(name="test")
    async def test_n8n_connection(self, ctx: commands.Context, *, test_feedback: str):
        """Test the n8n webhook connection."""
        conf = await self.config.guild(ctx.guild).all()
        
        if not conf["enabled"] or not conf["n8n_webhook_url"]:
            return await ctx.send("❌ KnowledgeBot is not properly configured.")
        
        processing_embed = discord.Embed(
            title="Testing n8n Connection",
            description="🤖 Sending test feedback to n8n...",
            color=discord.Color.blue()
        )
        processing_msg = await ctx.send(embed=processing_embed)
        
        try:
            ok = await self.send_to_n8n(ctx, test_feedback, conf["n8n_webhook_url"])

            if ok:
                result_embed = discord.Embed(
                    title="n8n Connection Test Successful",
                    description="✅ Webhook accepted by n8n. The workflow should post a reply.",
                    color=discord.Color.green()
                )
                result_embed.add_field(
                    name="Test Feedback",
                    value=test_feedback,
                    inline=False
                )
            else:
                result_embed = discord.Embed(
                    title="n8n Connection Test Failed",
                    description="❌ Failed to send test feedback to n8n.",
                    color=discord.Color.red()
                )
                result_embed.add_field(
                    name="Possible Issues",
                    value="• Check webhook URL\n• Verify n8n workflow is active\n• Check network connectivity",
                    inline=False
                )
            
            await processing_msg.edit(embed=result_embed)
            
        except Exception as e:
            log.error(f"Error in n8n test: {e}", exc_info=True)
            error_embed = discord.Embed(
                title="Test Error",
                description="❌ An error occurred during testing.",
                color=discord.Color.red()
            )
            await processing_msg.edit(embed=error_embed)
