import { prisma } from "@openviktor/db";
import { PipedreamClient } from "@openviktor/integrations";
import type { PipedreamConfig } from "@openviktor/integrations";
import { createLogger, isManaged, loadConfig } from "@openviktor/shared";
import {
	ConcurrencyExceededError,
	type Logger,
	ThreadLockedError,
	chunkMessage,
} from "@openviktor/shared";
import {
	LocalToolBackend,
	ModalToolBackend,
	ToolGatewayClient,
	connectIntegrationDefinition,
	createConnectIntegrationExecutor,
	createDisconnectIntegrationExecutor,
	createIntegrationSyncHandler,
	createListAvailableIntegrationsExecutor,
	createListWorkspaceConnectionsExecutor,
	createNativeRegistry,
	createSubmitPermissionRequestExecutor,
	createSyncWorkspaceConnectionsExecutor,
	deploySdkToWorkspace,
	disconnectIntegrationDefinition,
	listAvailableIntegrationsDefinition,
	listWorkspaceConnectionsDefinition,
	registerDbTools,
	registerSpacesTools,
	restoreToolsFromDb,
	submitPermissionRequestDefinition,
	syncWorkspaceConnectionsDefinition,
} from "@openviktor/tools";
import type { RegistryConfig, ToolBackend } from "@openviktor/tools";
import { ConvexClient, SpacesService, VercelClient } from "@openviktor/tools/spaces";
import { LLMGateway } from "./agent/gateway.js";
import { AnthropicProvider } from "./agent/providers/anthropic.js";
import { AgentRunner } from "./agent/runner.js";
import { createCronJobDefinition, createCronToolExecutors, createScriptCronDefinition, CronScheduler, deleteCronJobDefinition, listCronJobsDefinition, triggerCronJobDefinition } from "./cron/index.js";
import { buildOnboardingPrompt, isOnboardingNeeded, markOnboardingComplete, seedChannelIntros } from "./cron/onboarding.js";
import { IntegrationWatcher } from "./integrations/watcher.js";
import { seedBuiltinSkills } from "./skills/seed.js";
import { TelegramApi, type TelegramMessage, type TelegramUser } from "./telegram/api.js";
import { createConcurrencyLimiter } from "./thread/concurrency.js";
import { fetchActiveThreads } from "./thread/index.js";
import { ThreadLock } from "./thread/lock.js";
import { StaleThreadDetector } from "./thread/stale.js";
import { createDashboardApi } from "./tool-gateway/dashboard-api.js";
import { createToolGateway, registerWorkspaceToken } from "./tool-gateway/server.js";
import { UsageLimiter } from "./usage/limiter.js";
import { UsageTracker } from "./usage/tracker.js";
import { createSpacesApi } from "./spaces/api.js";

const logger = createLogger("bot");

function createToolBackend(config: ReturnType<typeof loadConfig>): {
	backend: ToolBackend;
	registry: ReturnType<typeof createNativeRegistry>;
} {
	let llmProvider: import("@openviktor/shared").LLMProvider | undefined;
	if (config.ANTHROPIC_API_KEY) {
		llmProvider = new AnthropicProvider(config.ANTHROPIC_API_KEY);
	}
	const registryConfig: RegistryConfig = {
		telegramToken: config.TELEGRAM_BOT_TOKEN,
		githubToken: config.GITHUB_TOKEN,
		browserbaseApiKey: config.BROWSERBASE_API_KEY,
		browserbaseProjectId: config.BROWSERBASE_PROJECT_ID,
		linkupApiKey: config.LINKUP_API_KEY,
		context7BaseUrl: config.CONTEXT7_BASE_URL,
		searchApiKey: config.SEARCH_API_KEY,
		imagenApiKey: config.IMAGEN_API_KEY,
		llmProvider,
		defaultModel: config.DEFAULT_MODEL,
	};
	const registry = createNativeRegistry(registryConfig);
	registerDbTools(registry, prisma);

	if (config.TOOL_BACKEND === "modal") {
		const backend = new ModalToolBackend({
			endpointUrl: config.MODAL_ENDPOINT_URL as string,
			authToken: config.MODAL_AUTH_TOKEN,
			timeoutMs: config.TOOL_TIMEOUT_MS,
		});
		logger.info({ endpoint: config.MODAL_ENDPOINT_URL }, "Using Modal tool backend");
		return { backend, registry };
	}

	const backend = new LocalToolBackend(registry);
	logger.info("Using local tool backend");
	return { backend, registry };
}

function getDisplayName(user?: TelegramUser): string | null {
	if (!user) return null;
	const fullName = [user.first_name, user.last_name].filter(Boolean).join(" ").trim();
	return fullName || user.username || String(user.id);
}

function normalizeIncomingText(message: TelegramMessage, botUsername?: string): string {
	const raw = (message.text ?? message.caption ?? "").trim();
	if (!raw || !botUsername) return raw;
	const mentionPattern = new RegExp(`@${botUsername}\\b`, "ig");
	return raw.replace(mentionPattern, "").replace(/\s{2,}/g, " ").trim() || raw;
}

function isTelegramSlashCommand(message: TelegramMessage, botUsername?: string): boolean {
	const raw = (message.text ?? message.caption ?? "").trim();
	if (!raw.startsWith("/")) return false;
	if (!botUsername) return true;
	const firstToken = raw.split(/\s+/, 1)[0] ?? raw;
	const commandWithoutSlash = firstToken.slice(1);
	const [commandName, commandTarget] = commandWithoutSlash.split("@", 2);
	if (!commandName) return false;
	return !commandTarget || commandTarget.toLowerCase() === botUsername.toLowerCase();
}

function normalizeSlashCommand(rawText: string, botUsername?: string): string {
	const trimmed = rawText.trim();
	if (!trimmed.startsWith("/")) return trimmed;

	const [firstToken, ...rest] = trimmed.split(/\s+/);
	const commandWithTarget = firstToken.slice(1);
	const [commandName] = commandWithTarget.split("@", 2);
	const args = rest.join(" ").trim();

	switch (commandName.toLowerCase()) {
		case "start":
			return "Introduce yourself, explain how to use you in this Telegram chat, and give 3 practical example prompts.";
		case "help":
			return "Explain how to use you in Telegram, including direct messages, group mentions, replies, tools, and integrations.";
		case "tools":
			return "List the tools and capabilities you support in this workspace. Group them by category and keep it concise.";
		case "status":
			return "Summarize your current status in this workspace, including connected integrations if any, available skills, and what you can help with right now.";
		case "new":
			return args
				? `Treat this as a fresh request with no dependency on earlier chat context: ${args}`
				: "Treat the next response as a fresh conversation starter and ask what the user wants help with.";
		default:
			return args ? `${commandName} ${args}` : commandName;
	}
}

function shouldHandleMessage(
	message: TelegramMessage,
	botProfile: TelegramUser,
	botUsername?: string,
): { handle: boolean; triggerType: "DM" | "MENTION"; threadTs: string } {
	const isDm = message.chat.type === "private";
	const isSlashCommand = isTelegramSlashCommand(message, botUsername);
	if (isDm) {
		return { handle: true, triggerType: "DM", threadTs: String(message.message_id) };
	}

	const text = message.text ?? message.caption ?? "";
	const mentioned = botUsername
		? new RegExp(`(^|\\s)@${botUsername}\\b`, "i").test(text)
		: false;
	const repliedToBot = message.reply_to_message?.from?.id === botProfile.id;

	if (!mentioned && !repliedToBot && !isSlashCommand) {
		return { handle: false, triggerType: "MENTION", threadTs: String(message.message_id) };
	}

	return {
		handle: true,
		triggerType: "MENTION",
		threadTs: String(message.reply_to_message?.message_id ?? message.message_id),
	};
}

async function resolveWorkspace(botProfile: TelegramUser, token: string) {
	const syntheticTeamId = `telegram:${botProfile.id}`;
	const workspaceName = botProfile.username ? `Telegram @${botProfile.username}` : "Telegram";

	const existing = await prisma.workspace.findUnique({
		where: { slackTeamId: syntheticTeamId },
	});
	if (existing) return existing;

	return prisma.workspace.create({
		data: {
			slackTeamId: syntheticTeamId,
			slackTeamName: workspaceName,
			slackBotToken: token,
			slackBotUserId: String(botProfile.id),
			settings: { transport: "telegram" },
		},
	});
}

async function resolveMember(workspaceId: string, user: TelegramUser) {
	return prisma.member.upsert({
		where: {
			workspaceId_slackUserId: {
				workspaceId,
				slackUserId: String(user.id),
			},
		},
		update: {
			displayName: getDisplayName(user),
			profile: {
				username: user.username ?? null,
				firstName: user.first_name ?? null,
				lastName: user.last_name ?? null,
				transport: "telegram",
			},
		},
		create: {
			workspaceId,
			slackUserId: String(user.id),
			displayName: getDisplayName(user),
			profile: {
				username: user.username ?? null,
				firstName: user.first_name ?? null,
				lastName: user.last_name ?? null,
				transport: "telegram",
			},
		},
	});
}

async function fetchSkillCatalog(workspaceId: string): Promise<string[]> {
	const skills = await prisma.skill.findMany({
		where: { workspaceId },
		select: { name: true, description: true, version: true },
		orderBy: { name: "asc" },
	});
	return skills.map((s) => {
		const desc = s.description ? ` - ${s.description}` : "";
		return `${s.name} (v${s.version})${desc}`;
	});
}

async function fetchIntegrationCatalog(workspaceId: string): Promise<string[]> {
	const skills = await prisma.skill.findMany({
		where: { workspaceId, name: { startsWith: "pd_" } },
		select: { name: true, description: true },
		orderBy: { name: "asc" },
	});
	return skills.map((s) => {
		const appName = s.name.replace(/^pd_/, "");
		const desc = s.description ?? appName;
		return `${appName}: ${desc}`;
	});
}

async function sendTelegramReply(
	api: TelegramApi,
	chatId: string,
	text: string,
	replyToMessageId?: string,
): Promise<void> {
	for (const chunk of chunkMessage(text, 4096).filter((entry) => entry.trim().length > 0)) {
		await api.sendMessage(chatId, chunk, replyToMessageId);
	}
}

async function main(): Promise<void> {
	const config = loadConfig();
	if (isManaged(config)) {
		throw new Error("Managed mode is not supported in the Telegram-first build.");
	}
	if (!config.TELEGRAM_BOT_TOKEN) {
		throw new Error("TELEGRAM_BOT_TOKEN is required.");
	}

	await prisma.$connect();
	logger.info({ mode: config.DEPLOYMENT_MODE }, "Database connected");

	const { backend, registry } = createToolBackend(config);
	const gateway = createToolGateway({
		registry,
		backend,
		logger: createLogger("tool-gateway"),
		defaultTimeoutMs: config.TOOL_TIMEOUT_MS,
	});

	const gatewayClient = new ToolGatewayClient({
		baseUrl: `http://localhost:${config.TOOL_GATEWAY_PORT}`,
		token: "local",
		timeoutMs: config.TOOL_TIMEOUT_MS,
	});
	registerWorkspaceToken("local", "default");

	const concurrencyLimiter = await createConcurrencyLimiter(
		config.MAX_CONCURRENT_RUNS,
		createLogger("concurrency"),
		config.REDIS_URL,
		config.AGENT_TIMEOUT_MS,
	);

	const threadLock = new ThreadLock(
		prisma,
		createLogger("thread-lock"),
		config.THREAD_LOCK_TIMEOUT_MS,
	);

	const staleDetector = new StaleThreadDetector(
		prisma,
		createLogger("stale-detector"),
		config.STALE_THREAD_TIMEOUT_MS,
		config.STALE_CHECK_INTERVAL_MS,
	);
	staleDetector.start();

	const usageTracker = new UsageTracker(prisma, createLogger("usage-tracker"));
	const usageLimiter = new UsageLimiter(
		prisma,
		createLogger("usage-limiter"),
		config.GLOBAL_MONTHLY_BUDGET_CENTS,
	);

	const llm = new LLMGateway(config);
	const runner = new AgentRunner(
		prisma,
		llm,
		createLogger("agent-runner"),
		{
			client: gatewayClient,
			tools: registry.getDefinitions(),
		},
		{
			concurrencyLimiter,
			threadLock,
			maxConcurrentRuns: config.MAX_CONCURRENT_RUNS,
		},
	);

	const scheduler = new CronScheduler(prisma, runner, createLogger("cron-scheduler"), {
		checkIntervalMs: config.CRON_CHECK_INTERVAL_MS,
		heartbeatEnabled: config.HEARTBEAT_ENABLED,
		slackToken: config.TELEGRAM_BOT_TOKEN,
		defaultModel: config.DEFAULT_MODEL,
		encryptionKey: config.ENCRYPTION_KEY,
		backend,
	});

	const cronTools = createCronToolExecutors(prisma, scheduler);
	const local = { localOnly: true };
	registry.register("create_cron_job", createCronJobDefinition, cronTools.create_cron_job, local);
	registry.register("create_script_cron", createScriptCronDefinition, cronTools.create_script_cron, local);
	registry.register("delete_cron_job", deleteCronJobDefinition, cronTools.delete_cron_job, local);
	registry.register("trigger_cron_job", triggerCronJobDefinition, cronTools.trigger_cron_job, local);
	registry.register("list_cron_jobs", listCronJobsDefinition, cronTools.list_cron_jobs, local);

	let spacesApi: ReturnType<typeof createSpacesApi> | undefined;
	if (config.CONVEX_ACCESS_TOKEN && config.CONVEX_TEAM_ID && config.VERCEL_TOKEN) {
		const convexClient = new ConvexClient({
			accessToken: config.CONVEX_ACCESS_TOKEN,
			teamId: config.CONVEX_TEAM_ID,
		});
		const vercelClient = new VercelClient({
			token: config.VERCEL_TOKEN,
			orgId: config.VERCEL_ORG_ID ?? "",
			domain: config.SPACES_DOMAIN,
		});
		const spacesService = new SpacesService({
			prisma,
			convex: convexClient,
			vercel: vercelClient,
			spacesDir: "/data/workspaces",
			spacesApiUrl: config.BASE_URL,
		});
		registerSpacesTools(registry, spacesService);
		spacesApi = createSpacesApi({
			spacesService,
			registry,
			logger: createLogger("spaces-api"),
			defaultTimeoutMs: config.TOOL_TIMEOUT_MS,
			resendApiKey: config.RESEND_API_KEY,
		});
	}

	let integrationWatcher: IntegrationWatcher | undefined;
	let pdClient: PipedreamClient | undefined;
	let syncHandler: ReturnType<typeof createIntegrationSyncHandler> | undefined;
	const hasPipedream = !!(
		config.PIPEDREAM_CLIENT_ID &&
		config.PIPEDREAM_CLIENT_SECRET &&
		config.PIPEDREAM_PROJECT_ID
	);
	if (hasPipedream) {
		const pdConfig: PipedreamConfig = {
			clientId: config.PIPEDREAM_CLIENT_ID as string,
			clientSecret: config.PIPEDREAM_CLIENT_SECRET as string,
			projectId: config.PIPEDREAM_PROJECT_ID as string,
			environment: config.PIPEDREAM_ENVIRONMENT,
		};
		pdClient = new PipedreamClient(pdConfig);
		syncHandler = createIntegrationSyncHandler(
			registry,
			pdClient,
			prisma,
			config.DANGEROUSLY_SKIP_PERMISSIONS,
		);

		const refreshRunnerTools = () => {
			runner.updateToolConfig({
				client: gatewayClient,
				tools: registry.getDefinitions(),
			});
		};

		integrationWatcher = new IntegrationWatcher(
			pdClient,
			syncHandler,
			refreshRunnerTools,
			createLogger("integration-watcher"),
		);

		registry.register(
			"list_available_integrations",
			listAvailableIntegrationsDefinition,
			createListAvailableIntegrationsExecutor(pdClient),
			local,
		);
		registry.register(
			"list_workspace_connections",
			listWorkspaceConnectionsDefinition,
			createListWorkspaceConnectionsExecutor(prisma),
			local,
		);
		registry.register(
			"connect_integration",
			connectIntegrationDefinition,
			createConnectIntegrationExecutor(pdClient, (workspaceId, appSlug) => {
				integrationWatcher?.watch(workspaceId, appSlug);
			}),
			local,
		);
		registry.register(
			"disconnect_integration",
			disconnectIntegrationDefinition,
			createDisconnectIntegrationExecutor(syncHandler),
			local,
		);
		registry.register(
			"sync_workspace_connections",
			syncWorkspaceConnectionsDefinition,
			createSyncWorkspaceConnectionsExecutor(syncHandler),
			local,
		);
		registry.register(
			"submit_permission_request",
			submitPermissionRequestDefinition,
			createSubmitPermissionRequestExecutor(prisma),
			local,
		);

		await restoreToolsFromDb(registry, pdClient, prisma, config.DANGEROUSLY_SKIP_PERMISSIONS);
	}

	runner.updateToolConfig({
		client: gatewayClient,
		tools: registry.getDefinitions(),
	});

	const allTools = registry.getAllDefinitions();
	prisma.workspace
		.findMany({ select: { id: true } })
		.then(async (workspaces) => {
			for (const ws of workspaces) {
				await deploySdkToWorkspace(ws.id, allTools).catch(() => {});
			}
		})
		.catch((err) => logger.warn({ err }, "Failed to deploy SDK to workspaces"));

	scheduler.start();

	let dashboardApi: { fetch: (req: Request) => Promise<Response> } | undefined;
	if (config.ENABLE_DASHBOARD) {
		dashboardApi = createDashboardApi({
			config,
			prisma,
			pdClient,
			integrationWatcher,
			disconnectApp: syncHandler?.disconnectApp.bind(syncHandler),
			usageLimiter,
			logger: createLogger("dashboard-api"),
		});
	}

	const corsHeaders: Record<string, string> = {
		"Access-Control-Allow-Origin": "*",
		"Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
		"Access-Control-Allow-Headers": "Content-Type, Authorization, X-Workspace-Id",
	};

	const gatewayServer = Bun.serve({
		port: config.TOOL_GATEWAY_PORT,
		fetch: async (req: Request) => {
			const url = new URL(req.url, "http://localhost");

			if (req.method === "OPTIONS") {
				return new Response(null, { status: 204, headers: corsHeaders });
			}

			if (url.pathname.startsWith("/api/viktor-spaces/") && spacesApi) {
				const response = await spacesApi.fetch(req);
				for (const [key, value] of Object.entries(corsHeaders)) {
					response.headers.set(key, value);
				}
				return response;
			}

			if (url.pathname.startsWith("/api/") && dashboardApi) {
				const response = await dashboardApi.fetch(req);
				for (const [key, value] of Object.entries(corsHeaders)) {
					response.headers.set(key, value);
				}
				return response;
			}

			return gateway.fetch(req);
		},
	});

	const telegramApi = new TelegramApi(config.TELEGRAM_BOT_TOKEN, createLogger("telegram"));
	const botProfile = await telegramApi.getMe();
	const botUsername = config.TELEGRAM_BOT_USERNAME ?? botProfile.username;
	const workspace = await resolveWorkspace(botProfile, config.TELEGRAM_BOT_TOKEN);
	registerWorkspaceToken("local", workspace.id);

	let stopRequested = false;
	let nextOffset = 0;

	const processTelegramMessage = async (message: TelegramMessage): Promise<void> => {
		const text = message.text ?? message.caption ?? "";
		if (!text || !message.from || message.from.is_bot) return;

		const route = shouldHandleMessage(message, botProfile, botUsername);
		if (!route.handle) return;

		const chatId = String(message.chat.id);
		const threadTs = route.threadTs;
		const member = await resolveMember(workspace.id, message.from);
		const rawText = normalizeIncomingText(message, botUsername);
		const normalizedText = isTelegramSlashCommand(message, botUsername)
			? normalizeSlashCommand(rawText, botUsername)
			: rawText;

		const [skillCatalog, integrationCatalog, activeThreads] = await Promise.all([
			fetchSkillCatalog(workspace.id),
			fetchIntegrationCatalog(workspace.id),
			fetchActiveThreads(prisma, workspace.id),
		]);
		const onboardingNeeded = await isOnboardingNeeded(prisma, workspace);

		try {
			const budget = await usageLimiter.canRun(workspace.id);
			if (!budget.allowed) {
				const resetDate = new Date(budget.resetsAt).toLocaleDateString("en-US", {
					month: "long",
					day: "numeric",
				});
				await sendTelegramReply(
					telegramApi,
					chatId,
					`This workspace has reached its usage limit. Usage resets on ${resetDate}.`,
					String(message.message_id),
				);
				return;
			}

			const result = await runner.run({
				workspaceId: workspace.id,
				memberId: member.id,
				triggerType: onboardingNeeded ? "ONBOARDING" : route.triggerType,
				slackChannel: chatId,
				slackThreadTs: threadTs,
				userMessage: onboardingNeeded ? buildOnboardingPrompt(normalizedText) : normalizedText,
				promptContext: {
					workspaceName: workspace.slackTeamName,
					channel: chatId,
					slackThreadTs: threadTs,
					userMessageTs: String(message.message_id),
					triggerType: onboardingNeeded ? "ONBOARDING" : route.triggerType,
					userName: member.displayName ?? undefined,
					skillCatalog,
					integrationCatalog,
					activeThreads,
					...(onboardingNeeded
						? { onboardingPrompt: buildOnboardingPrompt(normalizedText) }
						: {}),
				},
			});

			if (onboardingNeeded) {
				await markOnboardingComplete(prisma, workspace);
				await seedChannelIntros(prisma, workspace.id, logger);
				await seedBuiltinSkills(prisma, workspace.id, logger);
			}

			void usageTracker.record(workspace.id, {
				inputTokens: result.inputTokens,
				outputTokens: result.outputTokens,
				costCents: result.costCents,
				toolExecutions: 0,
			});

			if (!result.messageSent && result.responseText.trim()) {
				await sendTelegramReply(telegramApi, chatId, result.responseText.trim(), threadTs);
			}
		} catch (error) {
			if (error instanceof ThreadLockedError) {
				runner.injectMessage(chatId, threadTs, normalizedText);
				return;
			}
			if (error instanceof ConcurrencyExceededError) {
				await sendTelegramReply(
					telegramApi,
					chatId,
					"I am handling several requests right now. Please try again in a moment.",
					String(message.message_id),
				);
				return;
			}
			logger.error({ err: error, chatId }, "Failed to handle Telegram message");
			await sendTelegramReply(
				telegramApi,
				chatId,
				"Something went wrong while processing your request. Please try again.",
				String(message.message_id),
			);
		}
	};

	const poll = async (): Promise<void> => {
		while (!stopRequested) {
			try {
				const updates = await telegramApi.getUpdates(nextOffset, 30);
				for (const update of updates) {
					nextOffset = update.update_id + 1;
					const message = update.message;
					if (!message) continue;
					await processTelegramMessage(message);
				}
			} catch (error) {
				logger.error({ err: error }, "Telegram polling failed");
				await new Promise((resolve) => setTimeout(resolve, 3000));
			}
		}
	};

	const pollingPromise = poll();

	const shutdown = async () => {
		stopRequested = true;
		logger.info("Shutting down");
		integrationWatcher?.stop();
		staleDetector.stop();
		scheduler.stop();
		await concurrencyLimiter.shutdown();
		gatewayServer.stop();
		await prisma.$disconnect();
		await pollingPromise.catch(() => {});
		process.exit(0);
	};

	process.on("SIGTERM", shutdown);
	process.on("SIGINT", shutdown);

	logger.info(
		{
			port: gatewayServer.port,
			workspaceId: workspace.id,
			telegramBotId: botProfile.id,
			telegramBotUsername: botUsername ?? null,
		},
		"OpenViktor started (telegram self-hosted)",
	);
}

main().catch((err) => {
	logger.error({ err }, "Fatal error during startup");
	process.exit(1);
});
