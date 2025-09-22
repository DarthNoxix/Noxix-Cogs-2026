# AbsenceManager Cog

A comprehensive absence management system for Discord servers that automatically maintains a central embed showing all current staff absences with beautiful UI components and automatic updates.

## Features

- 🎯 **Centralized Management**: Single embed that automatically updates when absences change
- ✨ **Beautiful UI**: Stunning embeds with emojis, colors, and professional formatting
- 🎨 **Interactive Components**: Beautiful buttons, modals, and user-friendly interfaces
- ⚡ **Automatic Updates**: Embed refreshes automatically when absences are added/removed
- 🔄 **Auto-cleanup**: Automatically removes expired absences (configurable)
- 🛡️ **Role-based Access**: Configurable authorized roles for absence management
- 📅 **Flexible Duration**: Support for specific dates, relative durations, and indefinite absences
- 🎭 **User-friendly**: Easy-to-use modals and buttons for all operations
- 💎 **Premium Design**: Professional appearance with beautiful formatting and emojis

## Setup

1. Load the cog: `[p]load AbsenceManager`
2. Set up the system: `[p]absence setup [channel]`
3. Configure authorized roles: `[p]absence role @RoleName`
4. Start managing absences!

**Note**: If you encounter any errors, try reloading the cog with `[p]reload AbsenceManager`

## Commands

### Setup Commands
- `[p]absence setup [channel]` - Set up the absence management system
- `[p]absence disable` - Disable the absence management system
- `[p]absence config` - View and modify configuration

### Management Commands
- `[p]absence add @user [reason]` - Add an absence for a user (opens modal)
- `[p]absence quickadd @user <duration> [reason]` - Quickly add an absence without modal
- `[p]absence remove @user` - Remove an absence for a user
- `[p]absence list` - View all current absences
- `[p]absence role @role` - Add/remove authorized roles

## Usage Examples

### Adding Absences

**Quick Method (Recommended):**
```
[p]absence quickadd @user for 5 days Vacation time
[p]absence quickadd @user until 2024-01-15 Medical leave
[p]absence quickadd @user indefinite Personal reasons
```

*Note: The absence start date is automatically set to when the command is executed. All timestamps are displayed in your local timezone using Discord's relative time format (e.g., "2 hours ago", "in 3 days").*

**Modal Method:**
When you use `[p]absence add @user`, a modal will open where you can:
- Enter a reason (optional)
- Specify duration using:
  - `until 2024-01-15` (specific date)
  - `for 5 days` (relative duration)
  - `for 2 weeks` (weeks)
  - `for 1 month` (months)
  - `indefinite` (no end date)

### Interactive Buttons
The absence embed includes buttons for:
- **⚡ Quick Add**: Opens quick add modal (user, duration, reason all in one)
- **🗑️ Remove Absence**: Opens user selection modal  
- **🔄 Refresh**: Manually refresh the embed

## Configuration Options

- **Authorized Roles**: Control who can manage absences
- **Auto-remove Expired**: Automatically remove expired absences
- **Grace Period**: Hours to wait before removing expired absences
- **Embed Title**: Customize the embed title
- **Show Avatars**: Include user avatars in the embed

## Permissions

- **Administrators**: Full access to all commands and configuration
- **Authorized Roles**: Can add/remove absences and use interactive buttons
- **Everyone**: Can view the absence embed

## Data Storage

The cog stores:
- Absence records (user ID, reason, dates, etc.)
- Guild configuration (channel, roles, settings)
- All data is stored locally and can be deleted by administrators

## Technical Details

- **Auto-cleanup**: Runs every hour to remove expired absences
- **Embed Updates**: Automatically triggered when absences change
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Performance**: Efficient data storage and retrieval

## Support

For issues or feature requests, please contact the cog author or create an issue in the repository.
