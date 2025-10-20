# DevAccessGiveaway

A Red-DiscordBot cog for creating free development access giveaways with embed messages and button interactions.

## Features

- 🎉 Create giveaways with custom descriptions
- ⏰ Set custom duration (seconds, minutes, hours, days, weeks)
- 🎯 Specify number of winners
- 🔘 Interactive button for users to enter
- 🏆 Automatic winner selection and role assignment
- 📊 Track participants and active giveaways
- ⚙️ Easy setup and configuration

## Setup

1. Load the cog: `[p]load DevAccessGiveaway`
2. Configure the giveaway channel and role: `[p]devgiveaway setup #channel @role`
3. Start creating giveaways!

## Commands

### Setup Commands
- `[p]devgiveaway setup <channel> <role>` - Set the channel where giveaways will be posted and the role to assign to winners
- `[p]devgiveaway config` - Show current configuration

### Giveaway Commands
- `[p]devgiveaway create <winners> <duration> [description]` - Create a new giveaway
- `[p]devgiveaway list` - List all active giveaways
- `[p]devgiveaway end <giveaway_id>` - Manually end a giveaway

## Examples

### Basic Giveaway
```
[p]devgiveaway create 5 1h Free development access for 5 lucky winners!
```

### Early Access Giveaway
```
[p]devgiveaway create 3 30m Early access to new features
```

### VIP Access Giveaway
```
[p]devgiveaway create 1 1d VIP development access
```

## Duration Formats

- `30s` - 30 seconds
- `5m` - 5 minutes
- `2h` - 2 hours
- `1d` - 1 day
- `1w` - 1 week

## How It Works

1. **Setup**: Configure the channel and role using the setup command
2. **Create**: Use the create command to start a giveaway
3. **Enter**: Users click the button to enter the giveaway
4. **Wait**: The bot automatically counts down the specified time
5. **Select**: Winners are randomly selected and assigned the role
6. **Notify**: The embed updates to show the winners

## Permissions

- **Admin/Manage Guild**: Required for all giveaway management commands
- **Manage Roles**: Required for the bot to assign roles to winners
- **Send Messages**: Required in the configured channel

## Notes

- Giveaways are stored per-guild
- The bot automatically handles timeouts and winner selection
- Users can only enter each giveaway once
- If there are fewer participants than winners, all participants win
- The bot will attempt to assign roles but won't error if it fails (e.g., due to permission issues)
