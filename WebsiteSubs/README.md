# Website Subs Cog

A comprehensive Discord bot cog for managing website subscription roles with tier-based access, automatic expiration, and subscription tracking.

## Features

- **Tier-based Role Management**: Support for Noble, Knight, Squire, Levy, and Farmer subscription tiers
- **Early Access Role**: Automatic early access role assignment for all subscribers
- **Automatic Expiration**: 30-day automatic role removal with background checking
- **Subscription Tracking**: Complete database storage of subscription data
- **Notification System**: Automated notifications with verification buttons
- **Manual Management**: Commands for adding current subscribers with custom dates
- **Verification System**: Interactive buttons for extending or removing subscriptions

## Setup

1. Load the cog: `[p]load websitesubs`
2. Run the setup command: `[p]websitesubs setup`

The setup command will automatically:
- Configure the Early Access role using predefined role ID (1239757904652013578)
- Configure tier-specific roles using predefined role IDs (Noble, Knight, Squire, Levy, Farmer)
- Create a notification channel for subscription tracking
- Set up all necessary permissions

## Commands

### Core Commands

- `[p]websitesubs setup` - Initial setup of the subscription system
- `[p]websitesubs give <member> [tier] [website_username]` - Give subscription roles to a member (default: farmer)
- `[p]websitesubs remove <member>` - Remove subscription roles from a member
- `[p]websitesubs addcurrent <member> <tier> [date] [website_username]` - Add a current subscriber with custom date
- `[p]websitesubs list` - List all active subscriptions
- `[p]websitesubs check <member>` - Check a member's subscription status
- `[p]websitesubs config` - Show current configuration
- `[p]websitesubs setroles <noble> <knight> <squire> <levy> <farmer>` - Manually set role IDs for each tier
- `[p]websitesubs setearlyaccess <role>` - Manually set the Early Access role
- `[p]websitesubs export` - Export all subscription data to a JSON file
- `[p]websitesubs import` - Import subscription data from a JSON file (attach file to message)

### Subscription Tiers

- `noble` - Noble tier subscription (highest)
- `knight` - Knight tier subscription
- `squire` - Squire tier subscription
- `levy` - Levy tier subscription
- `farmer` - Farmer tier subscription (lowest)

### Website Username Tracking

You can optionally include a website username when giving subscriptions to help verify if users are actually subscribed on your website:

```bash
# Give subscription with website username
[p]websitesubs give @user knight username123

# Add current subscriber with website username
[p]websitesubs addcurrent @user squire 2024-01-15 username123
```

The website username will be displayed in:
- Command confirmation messages
- Notification channel messages
- Subscription lists
- Individual subscription checks

This helps you verify if users are actually subscribed on your website when reviewing the notifications.

### Leave As Is Button

When using the `addcurrent` command, an additional "Leave As Is" button appears in the notification embed. This button allows you to:

- Acknowledge that you've added a current subscriber
- Let their subscription expire naturally without extending it
- Keep track of when their current subscription will expire
- Avoid accidentally extending subscriptions for users who may not be actively subscribed

This is particularly useful when adding existing subscribers who may have already paid for a specific period and you just want to track when their current subscription expires.

### Data Export/Import

The cog includes powerful export and import functionality for data migration and backup:

#### Export Data
```bash
[p]websitesubs export
```
- Exports all subscription data, configuration, and settings to a JSON file
- Includes guild information, role IDs, notification channel, and all subscriptions
- File is automatically named with timestamp and guild ID
- Perfect for backups or migrating to another bot

#### Import Data
```bash
[p]websitesubs import
```
- Attach a JSON export file to your message
- Shows a confirmation dialog with all import details
- **WARNING**: This will **OVERWRITE** all current data
- Includes safety checks and validation
- Interactive confirmation with "Confirm Import" and "Cancel" buttons

#### Use Cases
- **Backup**: Regular exports for data safety
- **Migration**: Moving to a different bot instance
- **Testing**: Importing test data to development servers
- **Recovery**: Restoring from a backup after data loss

## How It Works

### Subscription Process

1. **Give Subscription**: Use `[p]websitesubs give @user knight username123` to give a user subscription roles with their website username
2. **Automatic Tracking**: The system automatically tracks when the subscription was given, by whom, and when it expires
3. **Notification**: A message is sent to the notification channel with user details and verification buttons
4. **Verification**: Staff can click "Subscribed" to extend by 30 days, "Unsubscribed" to remove roles, or "Leave As Is" (for current subscribers) to let it expire naturally

### Automatic Expiration

- Subscriptions automatically expire after 30 days
- The bot checks for expired subscriptions every hour
- Expired users have their roles automatically removed
- Expired subscriptions are removed from the database

### Notification System

When a subscription is given, a notification is sent to the configured channel containing:
- User information (mention and ID)
- Subscription tier
- Who gave the subscription
- When it was given
- When it expires
- Website username (if provided)
- Interactive buttons for verification:
  - **Subscribed** (green) - Extends subscription by 30 days
  - **Unsubscribed** (red) - Removes subscription roles
  - **Leave As Is** (gray) - Only appears for current subscribers, lets subscription expire naturally

### Manual Management

For existing subscribers, use `[p]websitesubs addcurrent @user knight 2024-01-15 username123` to:
- Add them with a specific subscription date
- Set the correct expiration date (30 days from the given date)
- Send a notification for verification

## Configuration

The cog stores the following configuration per guild:
- Early Access role ID - Predefined ID: 1239757904652013578
- Tier role IDs (noble, knight, squire, levy, farmer) - Predefined IDs:
  - Noble: 1197718229838200904
  - Knight: 1197718231025188915
  - Squire: 1197718392338124903
  - Levy: 1197718366585106514
  - Farmer: 1197718231583039588
- Notification channel ID
- Subscription database (user ID → subscription data)

### Custom Role IDs

If you need to use different role IDs, you can use these commands:
```
[p]websitesubs setearlyaccess @EarlyAccessRole
[p]websitesubs setroles @NobleRole @KnightRole @SquireRole @LevyRole @FarmerRole
```

## Permissions

Required permissions:
- `manage_roles` - To add/remove subscription roles
- `send_messages` - To send notifications
- `read_messages` - To read command messages

## Data Storage

The cog stores:
- User subscription data (tier, given by, given at, expires at)
- Role configurations
- Channel configurations

All data is stored securely using Red's Config system and is automatically cleaned up when subscriptions expire.

## Troubleshooting

### Common Issues

1. **"Please run setup first"** - Run `[p]websitesubs setup` to configure roles and channels
2. **"I don't have permission"** - Ensure the bot has `manage_roles` permission
3. **Roles not being removed** - Check if the bot has permission to remove the specific roles
4. **Notifications not working** - Verify the notification channel exists and the bot can send messages there

### Reset Configuration

To reset the configuration:
1. Unload the cog: `[p]unload websitesubs`
2. Clear the config: `[p]config clear websitesubs`
3. Reload and setup again: `[p]load websitesubs` then `[p]websitesubs setup`

## Support

For issues or questions, please check the bot's logs and ensure all permissions are correctly set up.
