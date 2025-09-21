# Website Subs Cog

A comprehensive Discord bot cog for managing website subscription roles with tier-based access, automatic expiration, and subscription tracking.

## Features

- **Tier-based Role Management**: Support for Basic, Premium, Pro, and Enterprise subscription tiers
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
- Create or configure the Early Access role
- Create tier-specific roles (Basic Subscriber, Premium Subscriber, Pro Subscriber, Enterprise Subscriber)
- Create a notification channel for subscription tracking
- Set up all necessary permissions

## Commands

### Core Commands

- `[p]websitesubs setup` - Initial setup of the subscription system
- `[p]websitesubs give <member> [tier]` - Give subscription roles to a member (default: basic)
- `[p]websitesubs remove <member>` - Remove subscription roles from a member
- `[p]websitesubs addcurrent <member> <tier> [date]` - Add a current subscriber with custom date
- `[p]websitesubs list` - List all active subscriptions
- `[p]websitesubs check <member>` - Check a member's subscription status
- `[p]websitesubs config` - Show current configuration

### Subscription Tiers

- `basic` - Basic tier subscription
- `premium` - Premium tier subscription  
- `pro` - Pro tier subscription
- `enterprise` - Enterprise tier subscription

## How It Works

### Subscription Process

1. **Give Subscription**: Use `[p]websitesubs give @user premium` to give a user subscription roles
2. **Automatic Tracking**: The system automatically tracks when the subscription was given, by whom, and when it expires
3. **Notification**: A message is sent to the notification channel with user details and verification buttons
4. **Verification**: Staff can click "Subscribed" to extend by 30 days or "Unsubscribed" to remove roles

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
- Interactive buttons for verification

### Manual Management

For existing subscribers, use `[p]websitesubs addcurrent @user premium 2024-01-15` to:
- Add them with a specific subscription date
- Set the correct expiration date (30 days from the given date)
- Send a notification for verification

## Configuration

The cog stores the following configuration per guild:
- Early Access role ID
- Tier role IDs (basic, premium, pro, enterprise)
- Notification channel ID
- Subscription database (user ID → subscription data)

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
