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

    async def _get_all_cog_configs(self) -> Dict[str, Dict]:
        """
        Retrieve all configurations from all loaded cogs.
        
        Returns:
            Dict containing all cog configurations organized by cog name
        """
        all_configs = {}
        
        for cog_name, cog in self.bot.cogs.items():
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
        name: Optional[str] = None
    ):
        """
        Create a complete backup of all bot configurations.
        
        Args:
            name: Optional custom name for the backup file
        """
        if not name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"backup_{timestamp}"
        
        # Ensure name is safe for filename
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        
        backup_path = await self._get_backup_path()
        backup_file = backup_path / f"{safe_name}.json"
        
        if backup_file.exists():
            return await ctx.send(f"❌ A backup with the name `{safe_name}` already exists!")
        
        embed = discord.Embed(
            title="🔄 Creating Backup",
            description="Collecting all cog configurations...",
            color=discord.Color.blue()
        )
        embed.add_field(name="Backup Name", value=safe_name, inline=True)
        embed.add_field(name="Guilds", value=len(self.bot.guilds), inline=True)
        embed.add_field(name="Cogs", value=len(self.bot.cogs), inline=True)
        
        message = await ctx.send(embed=embed)
        
        try:
            # Get all configurations
            embed.description = "Retrieving configurations from all cogs..."
            await message.edit(embed=embed)
            
            all_configs = await self._get_all_cog_configs()
            
            # Create backup data
            embed.description = "Organizing backup data..."
            await message.edit(embed=embed)
            
            backup_data = {
                "metadata": await self._create_backup_metadata(),
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
            embed.add_field(
                name="Basic Information",
                value=f"**Created:** {metadata.get('created_at', 'Unknown')}\n"
                      f"**Bot:** {metadata.get('bot_name', 'Unknown')}\n"
                      f"**Guilds:** {metadata.get('guild_count', 'Unknown')}\n"
                      f"**Cogs:** {metadata.get('cog_count', 'Unknown')}",
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
        Upload a backup file through Discord.
        
        Attach a .json backup file to this command to upload it.
        """
        if not ctx.message.attachments:
            return await ctx.send("❌ Please attach a backup file (.json) to upload!")
        
        attachment = ctx.message.attachments[0]
        
        # Validate file
        if not attachment.filename.endswith('.json'):
            return await ctx.send("❌ Please upload a .json backup file!")
        
        if attachment.size > 25 * 1024 * 1024:  # 25MB limit
            return await ctx.send("❌ File too large! Maximum size is 25MB.")
        
        embed = discord.Embed(
            title="📤 Uploading Backup",
            description="Downloading and validating backup file...",
            color=discord.Color.blue()
        )
        message = await ctx.send(embed=embed)
        
        try:
            # Download the file
            backup_data = await attachment.read()
            
            # Validate JSON
            try:
                backup_json = json.loads(backup_data.decode('utf-8'))
            except json.JSONDecodeError as e:
                return await message.edit(
                    embed=discord.Embed(
                        title="❌ Invalid Backup File",
                        description=f"File is not valid JSON: {str(e)}",
                        color=discord.Color.red()
                    )
                )
            
            # Validate backup structure
            if 'metadata' not in backup_json or 'configurations' not in backup_json:
                return await message.edit(
                    embed=discord.Embed(
                        title="❌ Invalid Backup File",
                        description="File is not a valid BotBackup file (missing metadata or configurations)",
                        color=discord.Color.red()
                    )
                )
            
            # Generate filename
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
            
            # Success embed
            success_embed = discord.Embed(
                title="✅ Backup Uploaded Successfully",
                color=discord.Color.green()
            )
            success_embed.add_field(name="Filename", value=safe_filename, inline=True)
            success_embed.add_field(name="Size", value=f"{attachment.size / (1024*1024):.2f} MB", inline=True)
            success_embed.add_field(name="Original Bot", value=bot_name, inline=True)
            success_embed.add_field(name="Cogs", value=len(backup_json.get('configurations', {})), inline=True)
            success_embed.add_field(name="Guilds", value=metadata.get('guild_count', 'Unknown'), inline=True)
            success_embed.add_field(name="Created", value=created_at, inline=True)
            success_embed.timestamp = datetime.utcnow()
            
            await message.edit(embed=success_embed)
            
        except Exception as e:
            error_embed = discord.Embed(
                title="❌ Upload Failed",
                description=f"An error occurred while uploading the backup:\n```{str(e)}```",
                color=discord.Color.red()
            )
            await message.edit(embed=error_embed)
            log.error(f"Backup upload failed: {e}", exc_info=True)

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
            max_size = 7 * 1024 * 1024  # 7MB to be safe (Discord limit is 8MB for non-nitro)
            
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
            
            # Calculate how many cogs per part
            total_size = len(json.dumps(backup_data, separators=(',', ':')))
            estimated_parts = max(1, (total_size // max_size) + 1)
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
                current_part["configurations"][cog_name] = cog_config
                current_cogs += 1
                
                # Check if we need to start a new part
                part_size = len(json.dumps(current_part, separators=(',', ':')))
                if part_size > max_size or current_cogs >= cogs_per_part:
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
                
                # Create embed for this part
                part_embed = discord.Embed(
                    title=f"📥 Backup Part {i}/{total_parts}",
                    color=discord.Color.blue()
                )
                part_embed.add_field(name="Backup Name", value=backup_name, inline=True)
                part_embed.add_field(name="Part", value=f"{i}/{total_parts}", inline=True)
                part_embed.add_field(name="Cogs in Part", value=len(part["configurations"]), inline=True)
                part_embed.add_field(name="Size", value=f"{temp_file.stat().st_size / (1024*1024):.2f} MB", inline=True)
                
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
        
        # Find all parts for this backup
        part_files = list(backup_path.glob(f"{backup_name}_part_*_of_*.json"))
        
        if not part_files:
            return await ctx.send(f"❌ No split backup parts found for `{backup_name}`!")
        
        # Sort parts by part number
        def extract_part_number(filename):
            try:
                # Extract part number from filename like "backup_part_1_of_3.json"
                parts = filename.stem.split('_')
                part_idx = parts.index('part') + 1
                return int(parts[part_idx])
            except (ValueError, IndexError):
                return 0
        
        part_files.sort(key=extract_part_number)
        
        embed = discord.Embed(
            title="🔄 Restoring Split Backup",
            description=f"Found {len(part_files)} parts, reconstructing backup...",
            color=discord.Color.orange()
        )
        message = await ctx.send(embed=embed)
        
        try:
            # Reconstruct the full backup
            full_backup = None
            total_parts = 0
            
            for part_file in part_files:
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
                value=f"{len(part_files)} parts",
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
