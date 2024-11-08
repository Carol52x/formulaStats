# 🏁 formulaStats

Discord bot to view F1 stats. This is a fork of [SmCTwelve's f1-bot](https://github.com/SmCTwelve/f1-bot)




## Installation

The application requires **Python 3.11+**. 

```bash
git clone https://github.com/Carol52x/formulaStats.git
cd formulaStats/
poetry shell
pip install -r requirements --no-deps
```


**After install**:

Rename `.env.example` to `.env`

Fill in `.env` file and optionally, configure in `config.ini`

## Usage

To start the bot run `python -m main`. This will attempt to connect using the env Token (see installation above).

The console will display log messages according to the level specified in `config.ini` and also output to `logs/f1-bot.log`.

Edit `config.ini` to change message display behaviour or restrict the bot to certain Guilds - this will only sync slash commands to the listed servers rather than being globally available (note this will prevent commands being accessible via DM). There may be a delay or some commands missing as Discord syncs the commands to your server.

### Cache

The application relies on caching both to speed up command processing and to respect API limits. Additionally the FastF1 library includes its own data cache.

A `/cache` directory will be created in the project root when the bot is running. This may become large over time with session telemetry (~100 MB per race weekend). You can manually delete the `/cache` folder or specific subfolders, or a script is provided in the root directory: `python -m flushcache`. Make sure the bot is not running. A new cache will be created during the next startup.


# Commands



The bot uses Discord slash commands. Once the commands have synced with your Guild, they can be accessed with `/command-name`.

