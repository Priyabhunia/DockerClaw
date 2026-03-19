<p align="center">
  <img src="docs/assets/banner.jpg" alt="OpenViktor" width="100%"/>
</p>

<p align="center">
  <strong>An autonomous AI teammate for Telegram.</strong><br/>
  Open-source. Self-hosted. Extensible. MIT-licensed.<br/><br/>
  <a href="docs/self-hosting.md">Self-Hosting Guide</a> · <a href="https://github.com/zggf-zggf/openviktor/issues">Report a Bug</a>
</p>

## What is OpenViktor?

OpenViktor is a Telegram-first autonomous AI coworker. It reads messages, runs tools, learns from your workspace, and replies directly in chat while keeping the rest of the stack self-hostable.

## Features

- Telegram-native runtime for direct messages and group mentions/replies
- Multi-provider LLM engine with Claude, GPT, Gemini, and local OpenAI-compatible models
- Persistent memory and workspace skills
- Built-in tools for files, bash, git, docs lookup, browser automation, search, image generation, and integrations
- Admin dashboard for runs, threads, tools, usage, spaces, and integrations
- Pipedream-based integrations for thousands of external apps

## Quick Start

```bash
git clone https://github.com/zggf-zggf/openviktor.git
cd openviktor
bun install
bun run --filter @openviktor/bot setup
docker compose -f docker/docker-compose.selfhosted.yml up -d
```

You will need:

- [Bun](https://bun.sh) >= 1.2
- [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- At least one LLM provider key

## Connect Telegram

1. Open [@BotFather](https://t.me/BotFather) in Telegram.
2. Run `/newbot` and create your bot name and username.
3. Copy the bot token BotFather returns.
4. Run:

```bash
bun run --filter @openviktor/bot setup
```

5. Paste the Telegram bot token into the setup wizard.
6. If you want to use Ollama instead of a cloud model, leave the cloud API keys blank and set an Ollama model when prompted.
7. Start OpenViktor:

```bash
docker compose -f docker/docker-compose.selfhosted.yml up -d
```

8. Open Telegram and send your bot a direct message like `hello`.
9. Check logs if needed:

```bash
docker compose -f docker/docker-compose.selfhosted.yml logs -f bot
```

If you want the bot to answer in a group, add it to the group and either mention it with `@your_bot_username` or send a `/command` to it.

## Supported Tools

OpenViktor currently supports these built-in tool categories:

- Telegram messaging: send, edit, and delete Telegram bot messages
- Workspace operations: read, write, edit, glob, grep, and convert files
- Shell execution: sandboxed bash commands with time limits
- Git and GitHub: local git operations and GitHub CLI workflows
- Documentation and search: library docs lookup and AI-assisted search
- Browser automation: remote browser sessions and file downloads
- Memory and skills: read/write learnings and workspace skills
- Integrations: app discovery, connection management, and synced third-party tools
- Spaces: create and manage deployable app workspaces
- Media: image generation and local image viewing

Telegram command aliases are also supported. `/start`, `/help`, `/tools`, `/status`, `/new`, and any other `/<command>` message are routed through the same agent flow instead of a separate limited command handler.

## Architecture

```text
openviktor/
|- apps/
|  |- bot/        # Telegram bot + agent runtime
|  |- web/        # Admin dashboard
|  `- landing/    # Marketing site
|- packages/
|  |- db/         # PostgreSQL schema (Prisma)
|  |- shared/     # Types, config, logger, errors
|  |- tools/      # Tool registry + executors
|  `- integrations/
`- docker/
```

## Development

```bash
bun install
docker compose -f docker/docker-compose.yml up -d
bun run db:generate
bun run db:migrate
bun run dev
bun run test
bun run lint
bun run typecheck
```

## License

[MIT](LICENSE)



# Additional Details
if you are running the bot in Docker, code changes are not applied until you rebuild/restart the container
this repo copies source code into the Docker image during build, it does not live-read your local files
so after code changes, the right command is usually:
```
docker compose -f docker/docker-compose.selfhosted.yml up -d --build bot
```