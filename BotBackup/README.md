# BotBackup Cog

A comprehensive backup and restore system for Red bot configurations. This cog allows you to backup all cog configurations for every server and restore them with a single command, making bot migration seamless.

## Features

- **Complete Configuration Backup**: Backs up all cog configurations (global, guild, member, user)
- **Easy Migration**: Restore all configurations with a single command
- **Detailed Metadata**: Includes bot info, guild details, and backup statistics
- **Safe Operations**: Confirmation prompts for destructive operations
- **File Management**: List, view, and delete backup files
- **Configurable Settings**: Customizable backup location and retention policies

## Commands

### Core Commands

- `[p]backup create [name] [skip_cogs] [overwrite]` - Create a complete backup of all configurations
- `[p]backup restore <name> <confirm>` - Restore configurations from a backup
- `[p]backup list` - List all available backup files
- `[p]backup info <name>` - Get detailed information about a backup
- `[p]backup cogs` - List all available cogs that can be backed up or skipped

### Upload/Download Commands

- `[p]backup upload` - Upload a backup file through Discord (attach .json file)
- `[p]backup download <name>` - Download a backup file through Discord (auto-splits large files)
- `[p]backup restore-split <name> <confirm>` - Restore from split backup files

### Management Commands

- `[p]backup delete <name> <confirm>` - Delete a backup file
- `[p]backup cleanup` - Clean up old backup files based on retention settings

### Settings Commands

- `[p]backup settings` - View current backup settings
- `[p]backup setlocation <location>` - Set backup directory location
- `[p]backup setretention <days>` - Set backup retention period (0 = keep forever)

## Usage Examples

### Creating a Backup

```
[p]backup create
[p]backup create my_migration_backup
[p]backup create my_backup Assistant,Calculator True
[p]backup create my_backup Assistant True
```

**Parameters:**
- `name`: Custom name for the backup (optional)
- `skip_cogs`: Comma-separated list of cogs to skip (optional)
- `overwrite`: Set to `True` to overwrite existing backup with same name (optional)

### Restoring a Backup

```
[p]backup restore backup_20241201_143022 True
```

### Viewing Backup Information

```
[p]backup list
[p]backup info my_migration_backup
[p]backup cogs
```

### Uploading/Downloading Backups

```
[p]backup upload
# (attach a .json backup file to the message)

[p]backup download my_migration_backup
# (for large files, this will automatically split into multiple parts)
```

## Backup File Structure

Backup files are stored as JSON and contain:

```json
{
  "metadata": {
    "backup_version": "1.0",
    "created_at": "2024-12-01T14:30:22.123456",
    "bot_id": 123456789012345678,
    "bot_name": "MyBot",
    "guild_count": 5,
    "cog_count": 15,
    "guilds": [
      {
        "id": 987654321098765432,
        "name": "My Server",
        "member_count": 150,
        "owner_id": 111222333444555666
      }
    ]
  },
  "configurations": {
    "Assistant": {
      "global": { ... },
      "guilds": { ... },
      "members": { ... },
      "users": { ... }
    },
    "Calculator": {
      "guilds": { ... }
    }
  }
}
```

## Migration Process

### Method 1: Direct Upload/Download (Recommended)
1. **On Source Bot**: 
   - Run `[p]backup create migration_backup` (full backup)
   - Or `[p]backup create migration_backup Assistant True` (skip Assistant cog)
2. **Download Backup**: Run `[p]backup download migration_backup` and save the file(s)
   - For large backups (>7MB), multiple part files will be created
3. **On Target Bot**: 
   - For single file: Upload using `[p]backup upload` (attach the .json file)
   - For split files: Upload all part files using `[p]backup upload` (attach each part)
4. **Restore**: 
   - Single file: `[p]backup restore uploaded_[botname]_[timestamp] True`
   - Split files: `[p]backup restore-split migration_backup True`

### Method 2: Manual File Transfer
1. **On Source Bot**: Run `[p]backup create migration_backup`
2. **Copy Backup File**: Manually transfer the backup file from the bot's data directory
3. **On Target Bot**: Load the BotBackup cog and run `[p]backup restore migration_backup True`

## Safety Features

- **Confirmation Required**: Destructive operations require explicit confirmation
- **Validation**: Checks for backup file existence and validity
- **Error Handling**: Graceful handling of missing cogs or corrupted data
- **Detailed Logging**: Comprehensive logging for troubleshooting

## Requirements

- Red Bot 3.5.0+
- No additional dependencies required

## Permissions

- Owner only (all commands require bot owner permissions)

## Data Storage

Backup files are stored in the cog's data directory under the configured backup location (default: `backups/`). The cog respects Red's data management policies and can be safely unloaded/reloaded.

## Troubleshooting

### Common Issues

1. **"Cog not found" during restore**: The cog may not be loaded on the target bot
2. **Permission errors**: Ensure the bot has proper file system permissions
3. **Large backup files**: Consider cleaning up old backups regularly

### Logs

Check the bot logs for detailed error information. The cog logs all operations with the logger name `red.BotBackup`.

## Support

For issues or feature requests, please contact the cog author or create an issue in the repository.
