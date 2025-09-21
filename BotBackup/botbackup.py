import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import discord
from redbot.core import Config, commands
from redbot.core.bot import Red
from redbot.core.data_manager import cog_data_path

log = logging.getLogger("red.BotBackup")


class BotBackup(commands.Cog):
    """
    A comprehensive backup and restore system for Red bot configurations.
    
    This cog allows you to backup all cog configurations for every server
    and restore them with a single command, making bot migration seamless.
    """

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(
            self,
            identifier=0x4B4F5842,  # "NOX" in hex
            force_registration=True,
        )
        
        # Register default settings
        self.config.register_global(
            backup_location="backups",
            auto_backup=False,
            backup_retention_days=30,
        )

    async def cog_load(self):
        """Initialize the cog and create backup directory if needed."""
        backup_path = await self._get_backup_path()
        backup_path.mkdir(parents=True, exist_ok=True)
        log.info("BotBackup cog loaded successfully")

    async def _get_backup_path(self) -> Path:
        """Get the backup directory path."""
        backup_location = await self.config.backup_location()
        return cog_data_path(self) / backup_location

    async def _get_all_cog_configs(self, skip_cogs: List[str] = None) -> Dict[str, Dict]:
        """
        Retrieve all configurations from all loaded cogs.
        
        Args:
            skip_cogs: List of cog names to skip during backup
        
        Returns:
            Dict containing all cog configurations organized by cog name
        """
        if skip_cogs is None:
            skip_cogs = []
        
        all_configs = {}
        
        for cog_name, cog in self.bot.cogs.items():
            if cog_name in skip_cogs:
                log.info(f"Skipping cog '{cog_name}' as requested")
                continue
                
            if not hasattr(cog, 'config') or not isinstance(cog.config, Config):
                continue
                
            try:
                cog_configs = {}
                
                # Get global config
                try:
                    global_config = await cog.config.all()
                    if global_config:
                        cog_configs['global'] = global_config
                except Exception as e:
                    log.warning(f"Failed to get global config for {cog_name}: {e}")
                
                # Get guild configs
                try:
                    guild_configs = {}
                    for guild_id in await cog.config.all_guilds():
                        guild_data = await cog.config.guild_from_id(guild_id).all()
                        if guild_data:
                            guild_configs[str(guild_id)] = guild_data
                    if guild_configs:
                        cog_configs['guilds'] = guild_configs
                except Exception as e:
                    log.warning(f"Failed to get guild configs for {cog_name}: {e}")
                
                # Get member configs
                try:
                    member_configs = {}
                    for guild_id in await cog.config.all_guilds():
                        guild_members = {}
                        for member_id in await cog.config.member_from_ids(guild_id).all():
                            member_data = await cog.config.member_from_ids(guild_id, member_id).all()
                            if member_data:
                                guild_members[str(member_id)] = member_data
                        if guild_members:
                            member_configs[str(guild_id)] = guild_members
                    if member_configs:
                        cog_configs['members'] = member_configs
                except Exception as e:
                    log.warning(f"Failed to get member configs for {cog_name}: {e}")
                
                # Get user configs
                try:
                    user_configs = {}
                    for user_id in await cog.config.all_users():
                        user_data = await cog.config.user_from_id(user_id).all()
                        if user_data:
                            user_configs[str(user_id)] = user_data
                    if user_configs:
                        cog_configs['users'] = user_configs
                except Exception as e:
                    log.warning(f"Failed to get user configs for {cog_name}: {e}")
                
                if cog_configs:
                    all_configs[cog_name] = cog_configs
                    
            except Exception as e:
                log.error(f"Failed to backup config for cog {cog_name}: {e}")
                continue
        
        return all_configs

    async def _create_backup_metadata(self) -> Dict:
        """Create metadata for the backup."""
        return {
            "backup_version": "1.0",
            "created_at": datetime.utcnow().isoformat(),
            "bot_id": self.bot.user.id if self.bot.user else None,
            "bot_name": self.bot.user.name if self.bot.user else "Unknown",
            "guild_count": len(self.bot.guilds),
            "cog_count": len(self.bot.cogs),
            "guilds": [
                {
                    "id": guild.id,
                    "name": guild.name,
                    "member_count": guild.member_count,
                    "owner_id": guild.owner_id,
                }
                for guild in self.bot.guilds
            ],
        }

    @commands.group(name="backup")
    @commands.is_owner()
    async def backup_group(self, ctx: commands.Context):
        """Bot configuration backup and restore commands."""
        pass

    @backup_group.command(name="create")
    async def backup_create(
        self, 
        ctx: commands.Context, 
        name: Optional[str] = None,
        skip_cogs: Optional[str] = None,
        overwrite: bool = False
    ):
        """
        Create a complete backup of all bot configurations.
        
        Args:
            name: Optional custom name for the backup file
            skip_cogs: Comma-separated list of cog names to skip (e.g., "Assistant,Calculator")
            overwrite: Whether to overwrite an existing backup with the same name
        """
        if not name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"backup_{timestamp}"
        
        # Ensure name is safe for filename
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        
        backup_path = await self._get_backup_path()
        backup_file = backup_path / f"{safe_name}.json"
        
        if backup_file.exists() and not overwrite:
            return await ctx.send(
                f"❌ A backup with the name `{safe_name}` already exists!\n"
                f"Use `{ctx.prefix}backup create {safe_name} {skip_cogs or ''} True` to overwrite it."
            )
        
        # Parse skip_cogs
        skip_list = []
        if skip_cogs:
            skip_list = [cog.strip() for cog in skip_cogs.split(',') if cog.strip()]
            # Validate that the cogs exist
            valid_cogs = [cog for cog in skip_list if cog in self.bot.cogs]
            invalid_cogs = [cog for cog in skip_list if cog not in self.bot.cogs]
            
            if invalid_cogs:
                return await ctx.send(
                    f"❌ The following cogs don't exist and cannot be skipped: {', '.join(invalid_cogs)}\n"
                    f"Available cogs: {', '.join(list(self.bot.cogs.keys())[:10])}{'...' if len(self.bot.cogs) > 10 else ''}"
                )
        
        embed = discord.Embed(
            title="🔄 Creating Backup",
            description="Collecting all cog configurations...",
            color=discord.Color.blue()
        )
        embed.add_field(name="Backup Name", value=safe_name, inline=True)
        embed.add_field(name="Guilds", value=len(self.bot.guilds), inline=True)
        embed.add_field(name="Cogs", value=len(self.bot.cogs), inline=True)
        
        if skip_list:
            embed.add_field(name="Skipped Cogs", value=", ".join(skip_list), inline=False)
        
        if overwrite and backup_file.exists():
            embed.add_field(name="Overwrite", value="Yes", inline=True)
        
        message = await ctx.send(embed=embed)
        
        try:
            # Get all configurations
            embed.description = "Retrieving configurations from all cogs..."
            await message.edit(embed=embed)
            
            all_configs = await self._get_all_cog_configs(skip_list)
            
            # Create backup data
            embed.description = "Organizing backup data..."
            await message.edit(embed=embed)
            
            metadata = await self._create_backup_metadata()
            metadata["skipped_cogs"] = skip_list
            metadata["overwrite"] = overwrite
            
            backup_data = {
                "metadata": metadata,
                "configurations": all_configs,
            }
            
            # Write backup file
            embed.description = "Writing backup file..."
            await message.edit(embed=embed)
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            # Get file size
            file_size = backup_file.stat().st_size
            size_mb = file_size / (1024 * 1024)
            
            # Success embed
            success_embed = discord.Embed(
                title="✅ Backup Created Successfully",
                color=discord.Color.green()
            )
            success_embed.add_field(name="Backup Name", value=safe_name, inline=True)
            success_embed.add_field(name="File Size", value=f"{size_mb:.2f} MB", inline=True)
            success_embed.add_field(name="Cogs Backed Up", value=len(all_configs), inline=True)
            
            if skip_list:
                success_embed.add_field(name="Skipped Cogs", value=", ".join(skip_list), inline=False)
            
            success_embed.add_field(name="Location", value=f"`{backup_file}`", inline=False)
            success_embed.timestamp = datetime.utcnow()
            
            await message.edit(embed=success_embed)
            
        except Exception as e:
            error_embed = discord.Embed(
                title="❌ Backup Failed",
                description=f"An error occurred while creating the backup:\n```{str(e)}```",
                color=discord.Color.red()
            )
            await message.edit(embed=error_embed)
            log.error(f"Backup creation failed: {e}", exc_info=True)

    @backup_group.command(name="cogs")
    async def backup_cogs(self, ctx: commands.Context):
        """List all available cogs that can be backed up or skipped."""
        cogs_with_config = []
        cogs_without_config = []
        
        for cog_name, cog in self.bot.cogs.items():
            if hasattr(cog, 'config') and isinstance(cog.config, Config):
                cogs_with_config.append(cog_name)
            else:
                cogs_without_config.append(cog_name)
        
        embed = discord.Embed(
            title="📋 Available Cogs",
            color=discord.Color.blue()
        )
        
        if cogs_with_config:
            # Split into chunks if too many cogs
            cog_chunks = [cogs_with_config[i:i+10] for i in range(0, len(cogs_with_config), 10)]
            for i, chunk in enumerate(cog_chunks):
                field_name = f"Cogs with Config ({len(cogs_with_config)} total)" if i == 0 else f"Cogs with Config (continued)"
                embed.add_field(
                    name=field_name,
                    value="\n".join([f"• {cog}" for cog in chunk]),
                    inline=False
                )
        
        if cogs_without_config:
            embed.add_field(
                name=f"Cogs without Config ({len(cogs_without_config)} total)",
                value="\n".join([f"• {cog}" for cog in cogs_without_config[:10]]),
                inline=False
            )
            if len(cogs_without_config) > 10:
                embed.add_field(
                    name="Note",
                    value=f"... and {len(cogs_without_config) - 10} more cogs without config",
                    inline=False
                )
        
        embed.add_field(
            name="Usage",
            value=f"Use `{ctx.prefix}backup create <name> <skip_cogs> <overwrite>`\n"
                  f"Example: `{ctx.prefix}backup create my_backup Assistant,Calculator True`",
            inline=False
        )
        
        await ctx.send(embed=embed)

    @backup_group.command(name="list")
    async def backup_list(self, ctx: commands.Context):
        """List all available backup files."""
        backup_path = await self._get_backup_path()
        
        if not backup_path.exists():
            return await ctx.send("❌ No backup directory found!")
        
        backup_files = list(backup_path.glob("*.json"))
        
        if not backup_files:
            return await ctx.send("❌ No backup files found!")
        
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        embed = discord.Embed(
            title="📋 Available Backups",
            color=discord.Color.blue()
        )
        
        for i, backup_file in enumerate(backup_files[:10]):  # Show max 10
            try:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                
                metadata = backup_data.get('metadata', {})
                created_at = metadata.get('created_at', 'Unknown')
                guild_count = metadata.get('guild_count', 'Unknown')
                cog_count = metadata.get('cog_count', 'Unknown')
                
                # Format timestamp
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    formatted_time = created_at
                
                file_size = backup_file.stat().st_size / (1024 * 1024)
                
                embed.add_field(
                    name=f"{i+1}. {backup_file.stem}",
                    value=f"**Created:** {formatted_time}\n"
                          f"**Guilds:** {guild_count} | **Cogs:** {cog_count}\n"
                          f"**Size:** {file_size:.2f} MB",
                    inline=False
                )
                
            except Exception as e:
                embed.add_field(
                    name=f"{i+1}. {backup_file.stem} (Error)",
                    value=f"Could not read backup file: {str(e)}",
                    inline=False
                )
        
        if len(backup_files) > 10:
            embed.set_footer(text=f"Showing 10 of {len(backup_files)} backups")
        
        await ctx.send(embed=embed)

    @backup_group.command(name="info")
    async def backup_info(self, ctx: commands.Context, backup_name: str):
        """
        Get detailed information about a specific backup.
        
        Args:
            backup_name: Name of the backup file (without .json extension)
        """
        backup_path = await self._get_backup_path()
        backup_file = backup_path / f"{backup_name}.json"
        
        if not backup_file.exists():
            return await ctx.send(f"❌ Backup `{backup_name}` not found!")
        
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            metadata = backup_data.get('metadata', {})
            configurations = backup_data.get('configurations', {})
            
            embed = discord.Embed(
                title=f"📊 Backup Information: {backup_name}",
                color=discord.Color.blue()
            )
            
            # Basic info
            basic_info = f"**Created:** {metadata.get('created_at', 'Unknown')}\n"
            basic_info += f"**Bot:** {metadata.get('bot_name', 'Unknown')}\n"
            basic_info += f"**Guilds:** {metadata.get('guild_count', 'Unknown')}\n"
            basic_info += f"**Cogs:** {metadata.get('cog_count', 'Unknown')}"
            
            if metadata.get('skipped_cogs'):
                basic_info += f"\n**Skipped Cogs:** {', '.join(metadata['skipped_cogs'])}"
            
            if metadata.get('overwrite'):
                basic_info += f"\n**Overwrite:** Yes"
            
            embed.add_field(
                name="Basic Information",
                value=basic_info,
                inline=False
            )
            
            # Guild list
            guilds = metadata.get('guilds', [])
            if guilds:
                guild_list = []
                for guild in guilds[:5]:  # Show max 5 guilds
                    guild_list.append(f"• {guild['name']} ({guild['id']})")
                if len(guilds) > 5:
                    guild_list.append(f"... and {len(guilds) - 5} more")
                
                embed.add_field(
                    name="Guilds",
                    value="\n".join(guild_list),
                    inline=False
                )
            
            # Cog list
            if configurations:
                cog_list = []
                for cog_name, cog_config in list(configurations.items())[:10]:  # Show max 10 cogs
                    config_types = list(cog_config.keys())
                    cog_list.append(f"• {cog_name} ({', '.join(config_types)})")
                if len(configurations) > 10:
                    cog_list.append(f"... and {len(configurations) - 10} more")
                
                embed.add_field(
                    name="Cogs",
                    value="\n".join(cog_list),
                    inline=False
                )
            
            # File info
            file_size = backup_file.stat().st_size / (1024 * 1024)
            embed.add_field(
                name="File Information",
                value=f"**Size:** {file_size:.2f} MB\n"
                      f"**Location:** `{backup_file}`",
                inline=False
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Failed to read backup file: {str(e)}")

    @backup_group.command(name="upload")
    async def backup_upload(self, ctx: commands.Context):
        """
        Upload backup files through Discord.
        
        Attach one or more .json backup files to this command to upload them.
        Supports uploading multiple split backup parts at once.
        """
        if not ctx.message.attachments:
            return await ctx.send("❌ Please attach one or more backup files (.json) to upload!")
        
        attachments = ctx.message.attachments
        successful_uploads = []
        failed_uploads = []
        
        embed = discord.Embed(
            title="📤 Uploading Backup Files",
            description=f"Processing {len(attachments)} file(s)...",
            color=discord.Color.blue()
        )
        message = await ctx.send(embed=embed)
        
        for i, attachment in enumerate(attachments, 1):
            try:
                embed.description = f"Processing file {i}/{len(attachments)}: {attachment.filename}"
                await message.edit(embed=embed)
                
                result = await self._process_single_upload(attachment)
                if result['success']:
                    successful_uploads.append(result)
                else:
                    failed_uploads.append({'filename': attachment.filename, 'error': result['error']})
                    
            except Exception as e:
                failed_uploads.append({'filename': attachment.filename, 'error': str(e)})
                log.error(f"Failed to process {attachment.filename}: {e}")
        
        # Create final result embed
        if failed_uploads:
            color = discord.Color.orange() if successful_uploads else discord.Color.red()
            title = "⚠️ Upload Completed with Errors" if successful_uploads else "❌ Upload Failed"
        else:
            color = discord.Color.green()
            title = "✅ All Files Uploaded Successfully"
        
        result_embed = discord.Embed(
            title=title,
            color=color
        )
        
        result_embed.add_field(
            name="Summary",
            value=f"**Successful:** {len(successful_uploads)}\n**Failed:** {len(failed_uploads)}",
            inline=False
        )
        
        if successful_uploads:
            file_list = []
            for upload in successful_uploads:
                file_list.append(f"• {upload['filename']} ({upload['size']:.2f} MB)")
            result_embed.add_field(
                name="Successfully Uploaded",
                value="\n".join(file_list[:10]),  # Show max 10 files
                inline=False
            )
            if len(successful_uploads) > 10:
                result_embed.add_field(
                    name="Note",
                    value=f"... and {len(successful_uploads) - 10} more files",
                    inline=False
                )
        
        if failed_uploads:
            error_list = []
            for upload in failed_uploads:
                error_list.append(f"• {upload['filename']}: {upload['error']}")
            result_embed.add_field(
                name="Failed Uploads",
                value="\n".join(error_list[:5]),  # Show max 5 errors
                inline=False
            )
            if len(failed_uploads) > 5:
                result_embed.add_field(
                    name="Note",
                    value=f"... and {len(failed_uploads) - 5} more errors",
                    inline=False
                )
        
        result_embed.timestamp = datetime.utcnow()
        await message.edit(embed=result_embed)

    async def _process_single_upload(self, attachment):
        """Process a single file upload and return the result."""
        # Validate file
        if not attachment.filename.endswith('.json'):
            return {'success': False, 'error': 'Not a .json file'}
        
        if attachment.size > 25 * 1024 * 1024:  # 25MB limit
            return {'success': False, 'error': 'File too large (>25MB)'}
        
        try:
            # Download the file
            backup_data = await attachment.read()
            
            # Validate JSON
            try:
                backup_json = json.loads(backup_data.decode('utf-8'))
            except json.JSONDecodeError as e:
                return {'success': False, 'error': f'Invalid JSON: {str(e)}'}
            
            # Validate backup structure
            if 'metadata' not in backup_json and 'part_info' not in backup_json:
                return {'success': False, 'error': 'Not a valid BotBackup file'}
            
            # Check if this is a split backup part
            part_info = backup_json.get('part_info', {})
            backup_name = part_info.get('backup_name', '')
            
            if backup_name and part_info:
                # This is a split backup part, preserve the original filename
                current_part = part_info.get('current_part', 0)
                total_parts = part_info.get('total_parts', 0)
                subpart = part_info.get('subpart', 0)
                micro_part = part_info.get('micro_part', 0)
                total_subparts = part_info.get('total_subparts', 0)
                total_micro_parts = part_info.get('total_micro_parts', 0)
                
                if micro_part and total_micro_parts:
                    # Micro part
                    filename = f"{backup_name}_part_{current_part}_subpart_{subpart}_micro_{micro_part}_of_{total_micro_parts}.json"
                elif subpart and total_subparts:
                    # Subpart
                    filename = f"{backup_name}_part_{current_part}_subpart_{subpart}_of_{total_subparts}.json"
                elif current_part and total_parts:
                    # Regular part
                    filename = f"{backup_name}_part_{current_part}_of_{total_parts}.json"
                else:
                    # Fallback to original naming
                    filename = f"{backup_name}.json"
            else:
                # Regular backup, use original naming
                metadata = backup_json.get('metadata', {})
                created_at = metadata.get('created_at', 'unknown')
                bot_name = metadata.get('bot_name', 'unknown')
                
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    timestamp = dt.strftime("%Y%m%d_%H%M%S")
                except:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                filename = f"uploaded_{bot_name}_{timestamp}.json"
            
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
            
            # Save file
            backup_path = await self._get_backup_path()
            backup_file = backup_path / safe_filename
            
            # Check if file exists
            counter = 1
            original_name = safe_filename
            while backup_file.exists():
                name_part = original_name.rsplit('.', 1)[0]
                safe_filename = f"{name_part}_{counter}.json"
                backup_file = backup_path / safe_filename
                counter += 1
            
            with open(backup_file, 'wb') as f:
                f.write(backup_data)
            
            return {
                'success': True,
                'filename': safe_filename,
                'size': attachment.size / (1024*1024),
                'backup_name': backup_name or 'unknown'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    @backup_group.command(name="download")
    async def backup_download(self, ctx: commands.Context, backup_name: str):
        """
        Download a backup file through Discord.
        
        For large files (>8MB), the backup will be split into multiple parts.
        
        Args:
            backup_name: Name of the backup file (without .json extension)
        """
        backup_path = await self._get_backup_path()
        backup_file = backup_path / f"{backup_name}.json"
        
        if not backup_file.exists():
            return await ctx.send(f"❌ Backup `{backup_name}` not found!")
        
        try:
            file_size = backup_file.stat().st_size
            max_size = 5 * 1024 * 1024  # 5MB to be very safe (Discord limit is 8MB for non-nitro)
            
            # Create embed with file info
            try:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                
                metadata = backup_data.get('metadata', {})
                embed = discord.Embed(
                    title="📥 Backup Download",
                    color=discord.Color.blue()
                )
                embed.add_field(name="Filename", value=f"{backup_name}.json", inline=True)
                embed.add_field(name="Size", value=f"{file_size / (1024*1024):.2f} MB", inline=True)
                embed.add_field(name="Original Bot", value=metadata.get('bot_name', 'Unknown'), inline=True)
                embed.add_field(name="Cogs", value=len(backup_data.get('configurations', {})), inline=True)
                embed.add_field(name="Guilds", value=metadata.get('guild_count', 'Unknown'), inline=True)
                embed.add_field(name="Created", value=metadata.get('created_at', 'Unknown'), inline=True)
                
            except Exception:
                embed = discord.Embed(
                    title="📥 Backup Download",
                    description=f"Downloading backup file: `{backup_name}.json`",
                    color=discord.Color.blue()
                )
                embed.add_field(name="Size", value=f"{file_size / (1024*1024):.2f} MB", inline=True)
            
            if file_size <= max_size:
                # File is small enough to send directly
                await ctx.send(embed=embed, file=discord.File(backup_file, filename=f"{backup_name}.json"))
            else:
                # File is too large, need to split it
                embed.description = f"File too large ({file_size / (1024*1024):.2f} MB), splitting into parts..."
                await ctx.send(embed=embed)
                
                await self._split_and_send_backup(ctx, backup_file, backup_name, max_size)
            
        except Exception as e:
            await ctx.send(f"❌ Failed to download backup: {str(e)}")
            log.error(f"Backup download failed: {e}", exc_info=True)

    async def _split_and_send_backup(self, ctx: commands.Context, backup_file: Path, backup_name: str, max_size: int):
        """Split a large backup file and send it in parts."""
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # Split configurations into chunks
            configurations = backup_data.get('configurations', {})
            cog_names = list(configurations.keys())
            
            # Calculate how many cogs per part (be more conservative)
            total_size = len(json.dumps(backup_data, separators=(',', ':')))
            estimated_parts = max(1, (total_size // max_size) + 2)  # Add extra buffer
            cogs_per_part = max(1, len(cog_names) // estimated_parts)
            
            parts = []
            current_part = {
                "metadata": backup_data.get('metadata', {}),
                "configurations": {},
                "part_info": {
                    "total_parts": 0,  # Will be updated after we know how many parts
                    "current_part": 0,
                    "backup_name": backup_name
                }
            }
            
            part_num = 1
            current_cogs = 0
            
            for cog_name, cog_config in configurations.items():
                # Check if adding this cog would make the part too large
                test_part = current_part.copy()
                test_part["configurations"][cog_name] = cog_config
                test_size = len(json.dumps(test_part, separators=(',', ':')))
                
                if test_size > (max_size * 0.8) and current_part["configurations"]:
                    # Current part is full, start a new one
                    current_part["part_info"]["current_part"] = part_num
                    parts.append(current_part.copy())
                    
                    # Start new part
                    current_part = {
                        "metadata": backup_data.get('metadata', {}),
                        "configurations": {},
                        "part_info": {
                            "total_parts": 0,  # Will be updated
                            "current_part": 0,
                            "backup_name": backup_name
                        }
                    }
                    part_num += 1
                    current_cogs = 0
                
                # Add the cog to current part
                current_part["configurations"][cog_name] = cog_config
                current_cogs += 1
                
                # Check if we've reached the cog limit per part
                if current_cogs >= cogs_per_part:
                    current_part["part_info"]["current_part"] = part_num
                    parts.append(current_part.copy())
                    
                    # Start new part
                    current_part = {
                        "metadata": backup_data.get('metadata', {}),
                        "configurations": {},
                        "part_info": {
                            "total_parts": 0,  # Will be updated
                            "current_part": 0,
                            "backup_name": backup_name
                        }
                    }
                    part_num += 1
                    current_cogs = 0
            
            # Add the last part if it has content
            if current_part["configurations"]:
                current_part["part_info"]["current_part"] = part_num
                parts.append(current_part)
            
            # Update total parts count
            total_parts = len(parts)
            for part in parts:
                part["part_info"]["total_parts"] = total_parts
            
            # Send each part
            for i, part in enumerate(parts, 1):
                part_filename = f"{backup_name}_part_{i}_of_{total_parts}.json"
                
                # Create temporary file
                temp_file = backup_file.parent / f"temp_{part_filename}"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(part, f, indent=2, ensure_ascii=False)
                
                # Check if this part is still too large
                part_size = temp_file.stat().st_size
                if part_size > 6 * 1024 * 1024:  # 6MB hard limit
                    # This part is still too large, need to split it further
                    temp_file.unlink()
                    await self._split_large_part(ctx, part, backup_name, i, total_parts)
                    continue
                
                # Create embed for this part
                part_embed = discord.Embed(
                    title=f"📥 Backup Part {i}/{total_parts}",
                    color=discord.Color.blue()
                )
                part_embed.add_field(name="Backup Name", value=backup_name, inline=True)
                part_embed.add_field(name="Part", value=f"{i}/{total_parts}", inline=True)
                part_embed.add_field(name="Cogs in Part", value=len(part["configurations"]), inline=True)
                part_embed.add_field(name="Size", value=f"{part_size / (1024*1024):.2f} MB", inline=True)
                
                if i == 1:
                    part_embed.add_field(
                        name="Instructions",
                        value="Download all parts and use `[p]backup restore-split <backup_name>` to restore",
                        inline=False
                    )
                
                # Send the file
                await ctx.send(
                    embed=part_embed,
                    file=discord.File(temp_file, filename=part_filename)
                )
                
                # Clean up temp file
                temp_file.unlink()
                
                # Small delay between parts to avoid rate limits
                if i < total_parts:
                    await asyncio.sleep(1)
            
        except Exception as e:
            await ctx.send(f"❌ Failed to split backup: {str(e)}")
            log.error(f"Backup splitting failed: {e}", exc_info=True)

    async def _split_large_part(self, ctx: commands.Context, part: dict, backup_name: str, part_num: int, total_parts: int):
        """Split a part that's still too large into smaller sub-parts."""
        try:
            configurations = part["configurations"]
            cog_names = list(configurations.keys())
            
            # Split cogs into smaller chunks
            max_cogs_per_subpart = max(1, len(cog_names) // 2)  # Split in half
            
            subpart_num = 1
            current_subpart = {
                "metadata": part["metadata"],
                "configurations": {},
                "part_info": {
                    "total_parts": total_parts,
                    "current_part": part_num,
                    "backup_name": backup_name,
                    "subpart": subpart_num,
                    "total_subparts": 0  # Will be updated
                }
            }
            
            subparts = []
            
            for cog_name, cog_config in configurations.items():
                # Check if this individual cog is too large
                cog_size = len(json.dumps({"test": cog_config}, separators=(',', ':')))
                
                if cog_size > 1.5 * 1024 * 1024:  # 1.5MB limit for individual cogs
                    # This cog is too large, split it into micro-parts
                    await self._split_individual_cog(ctx, cog_name, cog_config, backup_name, part_num, total_parts, subpart_num)
                    subpart_num += 1
                    continue
                
                # Check if adding this cog would make the subpart too large
                test_subpart = current_subpart.copy()
                test_subpart["configurations"][cog_name] = cog_config
                test_size = len(json.dumps(test_subpart, separators=(',', ':')))
                
                if test_size > 2 * 1024 * 1024 and current_subpart["configurations"]:  # 2MB limit for subparts
                    # Current subpart is full, start a new one
                    current_subpart["part_info"]["subpart"] = subpart_num
                    subparts.append(current_subpart.copy())
                    
                    # Start new subpart
                    current_subpart = {
                        "metadata": part["metadata"],
                        "configurations": {},
                        "part_info": {
                            "total_parts": total_parts,
                            "current_part": part_num,
                            "backup_name": backup_name,
                            "subpart": 0,
                            "total_subparts": 0
                        }
                    }
                    subpart_num += 1
                
                # Add the cog to current subpart
                current_subpart["configurations"][cog_name] = cog_config
            
            # Add the last subpart if it has content
            if current_subpart["configurations"]:
                current_subpart["part_info"]["subpart"] = subpart_num
                subparts.append(current_subpart)
            
            # Update total subparts count
            total_subparts = len(subparts)
            for subpart in subparts:
                subpart["part_info"]["total_subparts"] = total_subparts
            
            # Send each subpart
            for i, subpart in enumerate(subparts, 1):
                subpart_filename = f"{backup_name}_part_{part_num}_subpart_{i}_of_{total_subparts}.json"
                
                # Create temporary file
                backup_path = await self._get_backup_path()
                temp_file = backup_path / f"temp_{subpart_filename}"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(subpart, f, indent=2, ensure_ascii=False)
                
                # Create embed for this subpart
                subpart_embed = discord.Embed(
                    title=f"📥 Backup Part {part_num} Subpart {i}/{total_subparts}",
                    color=discord.Color.orange()
                )
                subpart_embed.add_field(name="Backup Name", value=backup_name, inline=True)
                subpart_embed.add_field(name="Part", value=f"{part_num}/{total_parts}", inline=True)
                subpart_embed.add_field(name="Subpart", value=f"{i}/{total_subparts}", inline=True)
                subpart_embed.add_field(name="Cogs in Subpart", value=len(subpart["configurations"]), inline=True)
                subpart_embed.add_field(name="Size", value=f"{temp_file.stat().st_size / (1024*1024):.2f} MB", inline=True)
                
                if i == 1:
                    subpart_embed.add_field(
                        name="Instructions",
                        value="Download all parts and subparts, then use `[p]backup restore-split <backup_name>` to restore",
                        inline=False
                    )
                
                # Send the file
                await ctx.send(
                    embed=subpart_embed,
                    file=discord.File(temp_file, filename=subpart_filename)
                )
                
                # Clean up temp file
                temp_file.unlink()
                
                # Small delay between subparts
                if i < total_subparts:
                    await asyncio.sleep(1)
                    
        except Exception as e:
            await ctx.send(f"❌ Failed to split large part: {str(e)}")
            log.error(f"Large part splitting failed: {e}", exc_info=True)

    async def _split_individual_cog(self, ctx: commands.Context, cog_name: str, cog_config: dict, backup_name: str, part_num: int, total_parts: int, subpart_num: int):
        """Split an individual cog that's too large into micro-parts."""
        try:
            # Split the cog configuration by type (global, guilds, members, users)
            micro_parts = []
            current_micro = {
                "metadata": {"cog_name": cog_name, "backup_name": backup_name},
                "configurations": {cog_name: {}},
                "part_info": {
                    "total_parts": total_parts,
                    "current_part": part_num,
                    "backup_name": backup_name,
                    "subpart": subpart_num,
                    "micro_part": 0,
                    "total_micro_parts": 0
                }
            }
            
            micro_part_num = 1
            
            # Process each configuration type separately
            for config_type in ['global', 'guilds', 'members', 'users']:
                if config_type not in cog_config:
                    continue
                
                config_data = cog_config[config_type]
                
                if config_type == 'global':
                    # Global config is usually small, add it to current micro part
                    current_micro["configurations"][cog_name][config_type] = config_data
                else:
                    # For guilds, members, users - split by entries
                    for key, value in config_data.items():
                        # Check if adding this entry would make the micro part too large
                        test_micro = current_micro.copy()
                        if config_type not in test_micro["configurations"][cog_name]:
                            test_micro["configurations"][cog_name][config_type] = {}
                        test_micro["configurations"][cog_name][config_type][key] = value
                        
                        test_size = len(json.dumps(test_micro, separators=(',', ':')))
                        
                        if test_size > 1 * 1024 * 1024 and current_micro["configurations"][cog_name]:  # 1MB limit for micro parts
                            # Current micro part is full, save it
                            current_micro["part_info"]["micro_part"] = micro_part_num
                            micro_parts.append(current_micro.copy())
                            
                            # Start new micro part
                            current_micro = {
                                "metadata": {"cog_name": cog_name, "backup_name": backup_name},
                                "configurations": {cog_name: {}},
                                "part_info": {
                                    "total_parts": total_parts,
                                    "current_part": part_num,
                                    "backup_name": backup_name,
                                    "subpart": subpart_num,
                                    "micro_part": 0,
                                    "total_micro_parts": 0
                                }
                            }
                            micro_part_num += 1
                        
                        # Add the entry to current micro part
                        if config_type not in current_micro["configurations"][cog_name]:
                            current_micro["configurations"][cog_name][config_type] = {}
                        current_micro["configurations"][cog_name][config_type][key] = value
            
            # Add the last micro part if it has content
            if current_micro["configurations"][cog_name]:
                current_micro["part_info"]["micro_part"] = micro_part_num
                micro_parts.append(current_micro)
            
            # Update total micro parts count
            total_micro_parts = len(micro_parts)
            for micro_part in micro_parts:
                micro_part["part_info"]["total_micro_parts"] = total_micro_parts
            
            # Send each micro part
            for i, micro_part in enumerate(micro_parts, 1):
                micro_filename = f"{backup_name}_part_{part_num}_subpart_{subpart_num}_micro_{i}_of_{total_micro_parts}.json"
                
                # Create temporary file
                backup_path = await self._get_backup_path()
                temp_file = backup_path / f"temp_{micro_filename}"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(micro_part, f, indent=2, ensure_ascii=False)
                
                # Create embed for this micro part
                micro_embed = discord.Embed(
                    title=f"📥 Backup Part {part_num} Subpart {subpart_num} Micro {i}/{total_micro_parts}",
                    color=discord.Color.red()
                )
                micro_embed.add_field(name="Backup Name", value=backup_name, inline=True)
                micro_embed.add_field(name="Cog", value=cog_name, inline=True)
                micro_embed.add_field(name="Part", value=f"{part_num}/{total_parts}", inline=True)
                micro_embed.add_field(name="Subpart", value=f"{subpart_num}", inline=True)
                micro_embed.add_field(name="Micro Part", value=f"{i}/{total_micro_parts}", inline=True)
                micro_embed.add_field(name="Size", value=f"{temp_file.stat().st_size / (1024*1024):.2f} MB", inline=True)
                
                if i == 1:
                    micro_embed.add_field(
                        name="Instructions",
                        value="Download all parts, subparts, and micro parts, then use `[p]backup restore-split <backup_name>` to restore",
                        inline=False
                    )
                
                # Send the file
                await ctx.send(
                    embed=micro_embed,
                    file=discord.File(temp_file, filename=micro_filename)
                )
                
                # Clean up temp file
                temp_file.unlink()
                
                # Small delay between micro parts
                if i < total_micro_parts:
                    await asyncio.sleep(1)
                    
        except Exception as e:
            await ctx.send(f"❌ Failed to split individual cog {cog_name}: {str(e)}")
            log.error(f"Individual cog splitting failed for {cog_name}: {e}", exc_info=True)

    @backup_group.command(name="restore-split")
    async def backup_restore_split(
        self, 
        ctx: commands.Context, 
        backup_name: str,
        confirm: bool = False
    ):
        """
        Restore bot configurations from split backup files.
        
        Upload all backup parts first, then use this command to restore.
        
        ⚠️ WARNING: This will overwrite existing configurations!
        
        Args:
            backup_name: Name of the backup (without _part_X_of_Y.json)
            confirm: Must be True to confirm the restore operation
        """
        if not confirm:
            return await ctx.send(
                "⚠️ **WARNING:** This will overwrite ALL existing configurations!\n"
                f"To confirm, run: `{ctx.prefix}backup restore-split {backup_name} True`"
            )
        
        backup_path = await self._get_backup_path()
        
        # Find all parts for this backup (including subparts and micro parts)
        part_files = list(backup_path.glob(f"{backup_name}_part_*_of_*.json"))
        subpart_files = list(backup_path.glob(f"{backup_name}_part_*_subpart_*_of_*.json"))
        micro_files = list(backup_path.glob(f"{backup_name}_part_*_subpart_*_micro_*_of_*.json"))
        
        if not part_files and not subpart_files and not micro_files:
            return await ctx.send(f"❌ No split backup parts found for `{backup_name}`!")
        
        # Combine all files
        all_files = part_files + subpart_files + micro_files
        
        # Sort parts by part number, subpart number, and micro part number
        def extract_part_number(filename):
            try:
                # Extract part number from filename like "backup_part_1_of_3.json" or "backup_part_1_subpart_1_of_2.json" or "backup_part_1_subpart_1_micro_1_of_3.json"
                parts = filename.stem.split('_')
                part_idx = parts.index('part') + 1
                part_num = int(parts[part_idx])
                
                # Check if it's a micro part
                if 'micro' in parts:
                    subpart_idx = parts.index('subpart') + 1
                    subpart_num = int(parts[subpart_idx])
                    micro_idx = parts.index('micro') + 1
                    micro_num = int(parts[micro_idx])
                    return (part_num, subpart_num, micro_num)
                # Check if it's a subpart
                elif 'subpart' in parts:
                    subpart_idx = parts.index('subpart') + 1
                    subpart_num = int(parts[subpart_idx])
                    return (part_num, subpart_num, 0)
                else:
                    return (part_num, 0, 0)
            except (ValueError, IndexError):
                return (0, 0, 0)
        
        all_files.sort(key=extract_part_number)
        
        embed = discord.Embed(
            title="🔄 Restoring Split Backup",
            description=f"Found {len(all_files)} parts/subparts/micro parts, reconstructing backup...",
            color=discord.Color.orange()
        )
        message = await ctx.send(embed=embed)
        
        try:
            # Reconstruct the full backup
            full_backup = None
            total_parts = 0
            
            for part_file in all_files:
                with open(part_file, 'r', encoding='utf-8') as f:
                    part_data = json.load(f)
                
                part_info = part_data.get('part_info', {})
                total_parts = part_info.get('total_parts', 0)
                current_part = part_info.get('current_part', 0)
                
                if full_backup is None:
                    # Initialize with first part
                    full_backup = {
                        "metadata": part_data.get('metadata', {}),
                        "configurations": {}
                    }
                
                # Merge configurations
                part_configs = part_data.get('configurations', {})
                full_backup["configurations"].update(part_configs)
                
                embed.description = f"Processing part {current_part}/{total_parts}..."
                await message.edit(embed=embed)
            
            if not full_backup:
                return await message.edit(
                    embed=discord.Embed(
                        title="❌ Restore Failed",
                        description="Could not reconstruct backup from parts",
                        color=discord.Color.red()
                    )
                )
            
            # Now restore the reconstructed backup
            embed.description = f"Restoring {len(full_backup['configurations'])} cogs..."
            await message.edit(embed=embed)
            
            restored_cogs = []
            failed_cogs = []
            
            # Restore each cog's configuration
            for cog_name, cog_config in full_backup['configurations'].items():
                try:
                    cog = self.bot.get_cog(cog_name)
                    if not cog or not hasattr(cog, 'config'):
                        failed_cogs.append(f"{cog_name} (cog not found)")
                        continue
                    
                    # Restore global config
                    if 'global' in cog_config:
                        for key, value in cog_config['global'].items():
                            await cog.config.set_raw(key, value=value)
                    
                    # Restore guild configs
                    if 'guilds' in cog_config:
                        for guild_id_str, guild_data in cog_config['guilds'].items():
                            guild_id = int(guild_id_str)
                            for key, value in guild_data.items():
                                await cog.config.guild_from_id(guild_id).set_raw(key, value=value)
                    
                    # Restore member configs
                    if 'members' in cog_config:
                        for guild_id_str, members in cog_config['members'].items():
                            guild_id = int(guild_id_str)
                            for member_id_str, member_data in members.items():
                                member_id = int(member_id_str)
                                for key, value in member_data.items():
                                    await cog.config.member_from_ids(guild_id, member_id).set_raw(key, value=value)
                    
                    # Restore user configs
                    if 'users' in cog_config:
                        for user_id_str, user_data in cog_config['users'].items():
                            user_id = int(user_id_str)
                            for key, value in user_data.items():
                                await cog.config.user_from_id(user_id).set_raw(key, value=value)
                    
                    restored_cogs.append(cog_name)
                    
                except Exception as e:
                    failed_cogs.append(f"{cog_name} ({str(e)})")
                    log.error(f"Failed to restore {cog_name}: {e}")
            
            # Create result embed
            if failed_cogs:
                color = discord.Color.orange()
                title = "⚠️ Restore Completed with Errors"
            else:
                color = discord.Color.green()
                title = "✅ Split Backup Restored Successfully"
            
            result_embed = discord.Embed(
                title=title,
                color=color
            )
            
            result_embed.add_field(
                name="Restored Cogs",
                value=f"{len(restored_cogs)} cogs restored successfully",
                inline=False
            )
            
            result_embed.add_field(
                name="Parts Processed",
                value=f"{len(all_files)} parts/subparts/micro parts",
                inline=True
            )
            
            if restored_cogs:
                cog_list = ", ".join(restored_cogs[:10])
                if len(restored_cogs) > 10:
                    cog_list += f" ... and {len(restored_cogs) - 10} more"
                result_embed.add_field(
                    name="Successfully Restored",
                    value=cog_list,
                    inline=False
                )
            
            if failed_cogs:
                result_embed.add_field(
                    name="Failed Cogs",
                    value=f"{len(failed_cogs)} cogs failed to restore",
                    inline=False
                )
                failed_list = ", ".join(failed_cogs[:5])
                if len(failed_cogs) > 5:
                    failed_list += f" ... and {len(failed_cogs) - 5} more"
                result_embed.add_field(
                    name="Failed Details",
                    value=failed_list,
                    inline=False
                )
            
            result_embed.timestamp = datetime.utcnow()
            await message.edit(embed=result_embed)
            
        except Exception as e:
            error_embed = discord.Embed(
                title="❌ Split Restore Failed",
                description=f"An error occurred while restoring the split backup:\n```{str(e)}```",
                color=discord.Color.red()
            )
            await message.edit(embed=error_embed)
            log.error(f"Split backup restore failed: {e}", exc_info=True)

    @backup_group.command(name="restore")
    async def backup_restore(
        self, 
        ctx: commands.Context, 
        backup_name: str,
        confirm: bool = False
    ):
        """
        Restore bot configurations from a backup file.
        
        ⚠️ WARNING: This will overwrite existing configurations!
        
        Args:
            backup_name: Name of the backup file (without .json extension)
            confirm: Must be True to confirm the restore operation
        """
        if not confirm:
            return await ctx.send(
                "⚠️ **WARNING:** This will overwrite ALL existing configurations!\n"
                f"To confirm, run: `{ctx.prefix}backup restore {backup_name} True`"
            )
        
        backup_path = await self._get_backup_path()
        backup_file = backup_path / f"{backup_name}.json"
        
        if not backup_file.exists():
            return await ctx.send(f"❌ Backup `{backup_name}` not found!")
        
        embed = discord.Embed(
            title="🔄 Restoring Backup",
            description="Loading backup file...",
            color=discord.Color.orange()
        )
        message = await ctx.send(embed=embed)
        
        try:
            # Load backup data
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            metadata = backup_data.get('metadata', {})
            configurations = backup_data.get('configurations', {})
            
            embed.description = f"Restoring {len(configurations)} cogs..."
            await message.edit(embed=embed)
            
            restored_cogs = []
            failed_cogs = []
            
            # Restore each cog's configuration
            for cog_name, cog_config in configurations.items():
                try:
                    cog = self.bot.get_cog(cog_name)
                    if not cog or not hasattr(cog, 'config'):
                        failed_cogs.append(f"{cog_name} (cog not found)")
                        continue
                    
                    # Restore global config
                    if 'global' in cog_config:
                        for key, value in cog_config['global'].items():
                            await cog.config.set_raw(key, value=value)
                    
                    # Restore guild configs
                    if 'guilds' in cog_config:
                        for guild_id_str, guild_data in cog_config['guilds'].items():
                            guild_id = int(guild_id_str)
                            for key, value in guild_data.items():
                                await cog.config.guild_from_id(guild_id).set_raw(key, value=value)
                    
                    # Restore member configs
                    if 'members' in cog_config:
                        for guild_id_str, members in cog_config['members'].items():
                            guild_id = int(guild_id_str)
                            for member_id_str, member_data in members.items():
                                member_id = int(member_id_str)
                                for key, value in member_data.items():
                                    await cog.config.member_from_ids(guild_id, member_id).set_raw(key, value=value)
                    
                    # Restore user configs
                    if 'users' in cog_config:
                        for user_id_str, user_data in cog_config['users'].items():
                            user_id = int(user_id_str)
                            for key, value in user_data.items():
                                await cog.config.user_from_id(user_id).set_raw(key, value=value)
                    
                    restored_cogs.append(cog_name)
                    
                except Exception as e:
                    failed_cogs.append(f"{cog_name} ({str(e)})")
                    log.error(f"Failed to restore {cog_name}: {e}")
            
            # Create result embed
            if failed_cogs:
                color = discord.Color.orange()
                title = "⚠️ Restore Completed with Errors"
            else:
                color = discord.Color.green()
                title = "✅ Restore Completed Successfully"
            
            result_embed = discord.Embed(
                title=title,
                color=color
            )
            
            result_embed.add_field(
                name="Restored Cogs",
                value=f"{len(restored_cogs)} cogs restored successfully",
                inline=False
            )
            
            if restored_cogs:
                cog_list = ", ".join(restored_cogs[:10])
                if len(restored_cogs) > 10:
                    cog_list += f" ... and {len(restored_cogs) - 10} more"
                result_embed.add_field(
                    name="Successfully Restored",
                    value=cog_list,
                    inline=False
                )
            
            if failed_cogs:
                result_embed.add_field(
                    name="Failed Cogs",
                    value=f"{len(failed_cogs)} cogs failed to restore",
                    inline=False
                )
                failed_list = ", ".join(failed_cogs[:5])
                if len(failed_cogs) > 5:
                    failed_list += f" ... and {len(failed_cogs) - 5} more"
                result_embed.add_field(
                    name="Failed Details",
                    value=failed_list,
                    inline=False
                )
            
            result_embed.timestamp = datetime.utcnow()
            await message.edit(embed=result_embed)
            
        except Exception as e:
            error_embed = discord.Embed(
                title="❌ Restore Failed",
                description=f"An error occurred while restoring the backup:\n```{str(e)}```",
                color=discord.Color.red()
            )
            await message.edit(embed=error_embed)
            log.error(f"Backup restore failed: {e}", exc_info=True)

    @backup_group.command(name="delete")
    async def backup_delete(
        self, 
        ctx: commands.Context, 
        backup_name: str,
        confirm: bool = False
    ):
        """
        Delete a backup file.
        
        Args:
            backup_name: Name of the backup file (without .json extension)
            confirm: Must be True to confirm the deletion
        """
        if not confirm:
            return await ctx.send(
                f"⚠️ To confirm deletion, run: `{ctx.prefix}backup delete {backup_name} True`"
            )
        
        backup_path = await self._get_backup_path()
        backup_file = backup_path / f"{backup_name}.json"
        
        if not backup_file.exists():
            return await ctx.send(f"❌ Backup `{backup_name}` not found!")
        
        try:
            backup_file.unlink()
            await ctx.send(f"✅ Backup `{backup_name}` deleted successfully!")
        except Exception as e:
            await ctx.send(f"❌ Failed to delete backup: {str(e)}")

    @backup_group.command(name="settings")
    async def backup_settings(self, ctx: commands.Context):
        """View and modify backup settings."""
        backup_location = await self.config.backup_location()
        auto_backup = await self.config.auto_backup()
        retention_days = await self.config.backup_retention_days()
        
        embed = discord.Embed(
            title="⚙️ Backup Settings",
            color=discord.Color.blue()
        )
        
        embed.add_field(name="Backup Location", value=backup_location, inline=True)
        embed.add_field(name="Auto Backup", value="Enabled" if auto_backup else "Disabled", inline=True)
        embed.add_field(name="Retention Days", value=retention_days, inline=True)
        
        await ctx.send(embed=embed)

    @backup_group.command(name="setlocation")
    async def backup_set_location(self, ctx: commands.Context, location: str):
        """
        Set the backup directory location.
        
        Args:
            location: Directory name for backups (relative to cog data path)
        """
        # Validate location name
        if not location.replace('_', '').replace('-', '').isalnum():
            return await ctx.send("❌ Location name can only contain letters, numbers, underscores, and hyphens!")
        
        await self.config.backup_location.set(location)
        
        # Create new directory
        new_path = cog_data_path(self) / location
        new_path.mkdir(parents=True, exist_ok=True)
        
        await ctx.send(f"✅ Backup location set to `{location}`")

    @backup_group.command(name="setretention")
    async def backup_set_retention(self, ctx: commands.Context, days: int):
        """
        Set backup retention period in days.
        
        Args:
            days: Number of days to keep backups (0 = keep forever)
        """
        if days < 0:
            return await ctx.send("❌ Retention days cannot be negative!")
        
        await self.config.backup_retention_days.set(days)
        await ctx.send(f"✅ Backup retention set to {days} days")

    @backup_group.command(name="cleanup")
    async def backup_cleanup(self, ctx: commands.Context):
        """Clean up old backup files based on retention settings."""
        retention_days = await self.config.backup_retention_days()
        
        if retention_days == 0:
            return await ctx.send("❌ Cleanup is disabled (retention set to 0 days)")
        
        backup_path = await self._get_backup_path()
        
        if not backup_path.exists():
            return await ctx.send("❌ No backup directory found!")
        
        backup_files = list(backup_path.glob("*.json"))
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        
        deleted_count = 0
        for backup_file in backup_files:
            if backup_file.stat().st_mtime < cutoff_time:
                try:
                    backup_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    log.error(f"Failed to delete {backup_file}: {e}")
        
        await ctx.send(f"✅ Cleaned up {deleted_count} old backup files")
