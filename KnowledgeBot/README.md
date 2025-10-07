# KnowledgeBot

An intelligent Discord bot that integrates with your existing n8n automation workflows to automatically categorize user feedback and route it to appropriate departments using AI analysis. This cog provides a Discord-native interface for your n8n-powered knowledge system.

## Features

- 🔗 **n8n Integration**: Works seamlessly with your existing n8n workflows
- 🤖 **AI-Powered Categorization**: Uses your n8n workflow's OpenAI GPT-4 analysis
- 📋 **Automatic Routing**: Routes feedback through n8n to appropriate department channels
- 🏢 **Three Department Types**:
  - **Customer Success**: For success stories and positive feedback
  - **IT Department**: For urgent issues and critical problems
  - **Helpdesk**: For general support tickets
- 📊 **Comprehensive Logging**: Tracks all feedback submissions
- ⚙️ **Flexible Configuration**: Easy setup through Discord commands
- 🧪 **Testing Tools**: Test the n8n connection before going live

## Installation

1. Download the KnowledgeBot folder to your Red-DiscordBot cogs directory
2. Load the cog: `[p]load KnowledgeBot`
3. Set up your n8n webhook URL: `[p]knowledgebot setup <your_n8n_webhook_url>`

## Quick Setup

### 1. Basic Configuration

```bash
# Set up with your n8n webhook URL
[p]knowledgebot setup https://your-n8n-instance.com/webhook/your-webhook-id

# Check status
[p]knowledgebot status
```

### 2. Configure Departments (Optional)

```bash
# Set up Customer Success department
[p]knowledgebot department set success_story #customer-success "Customer Success"

# Set up IT Department for urgent issues
[p]knowledgebot department set urgent_issue #it-department "IT Department"

# Set up Helpdesk for general tickets
[p]knowledgebot department set ticket #helpdesk "Helpdesk"
```

### 3. Set Submission Channel (Optional)

```bash
# Set where users can submit feedback
[p]knowledgebot submission #feedback-channel

# Set logging channel for tracking
[p]knowledgebot log #bot-logs
```

### 4. Test the Connection

```bash
# Test the n8n webhook connection
[p]knowledgebot test This is a test message to verify n8n integration
```

## Usage

### For Users

Users can submit feedback using the `[p]feedback` command:

```bash
[p]feedback I'm having trouble logging into the system, it keeps saying my password is wrong
[p]feedback The new update is amazing! Everything works so much faster now
[p]feedback URGENT: The entire system is down and customers can't place orders!
```

The bot will:
1. Send the feedback to your n8n webhook
2. n8n processes it with AI categorization
3. n8n routes it to the appropriate department channel
4. Provide confirmation to the user

### For Administrators

#### Configuration Commands

```bash
# View current status and configuration
[p]knowledgebot status

# Enable/disable the system
[p]knowledgebot disable

# Test the n8n connection
[p]knowledgebot test This is a test message to verify n8n integration

# Manage departments
[p]knowledgebot department set <type> <channel> [name]
[p]knowledgebot department disable <type>
```

#### Department Types

- `success_story`: For positive feedback and success stories
- `urgent_issue`: For critical problems requiring immediate attention
- `ticket`: For general support requests

## Configuration Options

### Department Settings

Each department can be configured with:
- **Channel**: Where categorized feedback will be sent
- **Name**: Display name for the department
- **Enabled/Disabled**: Toggle department routing

### System Settings

- **Auto-categorization**: Automatically categorize and route feedback
- **Submission Channel**: Restrict feedback submission to specific channels
- **Log Channel**: Track all categorizations for review

## How It Works

1. **User submits feedback** using `[p]feedback <message>`
2. **Bot sends data to n8n** via webhook with user and context information
3. **n8n processes the feedback** using your existing OpenAI GPT-4 workflow
4. **n8n categorizes and routes** based on your workflow logic:
   - Success stories → Customer Success
   - Urgent issues → IT Department  
   - General requests → Helpdesk
5. **n8n sends responses** to appropriate department channels via Discord webhooks
6. **Logging** records the feedback submission for review

## AI Categorization Logic

The AI uses the following criteria:

### Success Story
- Positive feedback and praise
- User appreciation and satisfaction
- Success stories and testimonials

### Urgent Issue
- Extreme dissatisfaction
- Critical problems with immediate impact
- System outages or major failures
- Security concerns

### Ticket (Default)
- General support requests
- Feature requests
- Minor issues
- Questions and inquiries

## Example Workflow

1. **User**: `[p]feedback I can't access my account, getting error 500`
2. **Bot**: Sends data to n8n webhook with user context
3. **n8n**: Processes with AI → Categorizes as "urgent-issue" → Routes to IT Department
4. **IT Channel**: Receives formatted message from n8n with user details and context
5. **User**: Gets confirmation that their feedback was sent for processing

## Troubleshooting

### Common Issues

**"KnowledgeBot is not enabled"**
- Run `[p]knowledgebot setup <webhook_url>` to configure

**"Failed to process feedback through n8n"**
- Check your n8n webhook URL is correct
- Verify your n8n workflow is active and running
- Try the test command: `[p]knowledgebot test <message>`

**"Cannot send message to department channel"**
- Check bot permissions in the department channel
- Verify the channel ID is correct
- Ensure the department is enabled

### Testing

Use the test command to verify your n8n connection:

```bash
[p]knowledgebot test I'm having trouble with the login system
[p]knowledgebot test The new features are amazing!
[p]knowledgebot test URGENT: Server is down!
```

## Permissions

- **Admin**: Full configuration access
- **Users**: Can submit feedback using `[p]feedback`
- **Bot**: Needs permissions to send messages and embeds in configured channels

## Data Storage

This cog stores:
- Guild configuration settings
- Department channel mappings
- No user data is permanently stored

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify your configuration with `[p]knowledgebot status`
3. Test with `[p]knowledgebot test <message>`

## n8n Integration Details

### Webhook Payload Format

The bot sends the following data to your n8n webhook:

```json
{
  "feedback": "User's feedback message",
  "user_id": 123456789,
  "username": "username",
  "discriminator": "1234",
  "guild_id": 987654321,
  "guild_name": "Server Name",
  "channel_id": 456789123,
  "channel_name": "general",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "message_id": 789123456
}
```

### Expected n8n Workflow

Your n8n workflow should:
1. Receive the webhook data
2. Extract the `feedback` field
3. Use OpenAI to categorize the feedback
4. Route to appropriate Discord webhooks based on category
5. Return a success response (HTTP 200)

## Version History

- **v1.0.0**: Initial release with n8n integration and Discord-native feedback submission
