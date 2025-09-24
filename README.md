## House of the Dragon RP Bot

A Discord bot cog that sets up a House of the Dragon themed roleplay hub in a channel, with buttons, modals, user selects, and slash commands to keep the fun flowing.

### Features

- **Persistent control panel**:
  - **Create/Update Character** modal
  - **View My Profile** button
  - **Draw RP Prompt** button
  - **House selector**
  - **Lore compendium** selector
  - **Duel opponent** user-select
- **Slash commands**:
  - `/hotd setup` — post the control panel in a channel
  - `/hotd panel_refresh` — refresh/recreate the panel if missing
  - `/hotd profile_view [user]` — view a profile
  - `/hotd profile_set` — set your profile with options
  - `/hotd profile_reset` — delete your profile
  - `/hotd prompt [topic] [public]` — draw a prompt
  - `/hotd duel @opponent` — post a duel result
- **JSON persistence** in `data/` for profiles and panel state

### Requirements

- Python 3.10+
- A Discord bot token with the following intents enabled:
  - Server Members (Privileged) — optional but recommended for better user info

Install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Set your token and run:

```bash
export DISCORD_TOKEN=YOUR_TOKEN_HERE
python bot.py
```

Invite your bot with permissions to send messages, embed links, and use application commands.

### Usage

1. In your server, run `/hotd setup` in the channel you want the RP hub.
2. Use the buttons/selects to create characters, choose houses, draw prompts, and challenge duels.
3. Use `/hotd panel_refresh` if the panel is deleted or to rebuild controls after a reboot.

### Data Files

- `data/hotd_profiles.json` — user profiles keyed by user ID
- `data/hotd_state.json` — per-guild panel message reference

You can change the data directory by setting `HOTD_DATA_DIR`.

### Notes

- The control panel view is registered as a persistent view at startup so the buttons/selects continue working after restarts.
- If you alter custom IDs in `cogs/hotd_rp.py`, update the persistent view registration accordingly.
