import type { PrismaClient } from "@openviktor/db";
import type { LLMProvider } from "@openviktor/shared";
import { type ToolExecutor, ToolRegistry } from "../registry.js";
import {
	type SpacesService,
	createSpacesToolExecutors,
	spacesToolDefinitions,
} from "../spaces/index.js";
import {
	aiStructuredOutputDefinition,
	createAiStructuredOutputExecutor,
} from "./ai-structured-output.js";
import { bashDefinition, bashExecutor } from "./bash.js";
import {
	browserCloseSessionDefinition,
	browserCreateSessionDefinition,
	browserDownloadFilesDefinition,
	createBrowserExecutors,
} from "./browser.js";
import { coworkerText2ImDefinition, createText2ImExecutor } from "./coworker-text2im.js";
import {
	createCustomApiIntegrationDefinition,
	createCustomApiIntegrationExecutor,
} from "./create-custom-api-integration.js";
import {
	createDocsExecutors,
	queryLibraryDocsDefinition,
	resolveLibraryIdDefinition,
} from "./docs.js";
import { fileEditDefinition, fileEditExecutor } from "./file-edit.js";
import { fileReadDefinition, fileReadExecutor } from "./file-read.js";
import { fileToMarkdownDefinition, fileToMarkdownExecutor } from "./file-to-markdown.js";
import { fileWriteDefinition, fileWriteExecutor } from "./file-write.js";
import { coworkerGitDefinition, coworkerGithubCliDefinition, createGitExecutors } from "./git.js";
import { globDefinition, globExecutor } from "./glob.js";
import { grepDefinition, grepExecutor } from "./grep.js";
import {
	createReadLearningsExecutor,
	createWriteLearningExecutor,
	readLearningsDefinition,
	writeLearningDefinition,
} from "./learnings.js";
import { createQuickAiSearchExecutor, quickAiSearchDefinition } from "./quick-ai-search.js";
import {
	createListSkillsExecutor,
	createReadSkillExecutor,
	createWriteSkillExecutor,
	listSkillsDefinition,
	readSkillDefinition,
	writeSkillDefinition,
} from "./skills.js";
import {
	coworkerGetSlackReactionsDefinition,
	coworkerInviteSlackUserToTeamDefinition,
	coworkerJoinSlackChannelsDefinition,
	coworkerLeaveSlackChannelsDefinition,
	coworkerListSlackChannelsDefinition,
	coworkerListSlackUsersDefinition,
	coworkerOpenSlackConversationDefinition,
	coworkerReportIssueDefinition,
	createSlackAdminExecutors,
} from "./slack-admin.js";
import {
	coworkerDeleteSlackMessageDefinition,
	coworkerDownloadFromSlackDefinition,
	coworkerSendSlackMessageDefinition,
	coworkerSlackHistoryDefinition,
	coworkerSlackReactDefinition,
	coworkerUpdateSlackMessageDefinition,
	coworkerUploadToSlackDefinition,
	createSlackToolExecutors,
} from "./slack-comms.js";
import {
	coworkerDeleteTelegramMessageDefinition,
	coworkerEditTelegramMessageDefinition,
	coworkerSendTelegramMessageDefinition,
	createTelegramToolExecutors,
} from "./telegram-comms.js";
import {
	type ThreadOrchestrationDeps,
	createCreateThreadExecutor,
	createGetPathInfoExecutor,
	createListRunningPathsExecutor,
	createSendMessageToThreadExecutor,
	createThreadDefinition,
	createWaitForPathsExecutor,
	getPathInfoDefinition,
	listRunningPathsDefinition,
	sendMessageToThreadDefinition,
	waitForPathsDefinition,
} from "./thread-orchestration.js";
import { viewImageDefinition, viewImageExecutor } from "./view-image.js";
import { workspaceTreeDefinition, workspaceTreeExecutor } from "./workspace-tree.js";

export interface RegistryConfig {
	slackToken?: string;
	telegramToken?: string;
	githubToken?: string;
	browserbaseApiKey?: string;
	browserbaseProjectId?: string;
	linkupApiKey?: string;
	context7BaseUrl?: string;
	searchApiKey?: string;
	imagenApiKey?: string;
	llmProvider?: LLMProvider;
	defaultModel?: string;
}

export function createNativeRegistry(config: RegistryConfig = {}): ToolRegistry {
	const registry = new ToolRegistry();

	registry.register("bash", bashDefinition, bashExecutor);
	registry.register("file_read", fileReadDefinition, fileReadExecutor);
	registry.register("file_write", fileWriteDefinition, fileWriteExecutor);
	registry.register("file_edit", fileEditDefinition, fileEditExecutor);
	registry.register("glob", globDefinition, globExecutor);
	registry.register("grep", grepDefinition, grepExecutor);
	registry.register("view_image", viewImageDefinition, viewImageExecutor);

	registry.register("file_to_markdown", fileToMarkdownDefinition, fileToMarkdownExecutor);

	if (config.llmProvider) {
		registry.register(
			"ai_structured_output",
			aiStructuredOutputDefinition,
			createAiStructuredOutputExecutor(config.llmProvider, config.defaultModel),
		);
		registry.register(
			"quick_ai_search",
			quickAiSearchDefinition,
			createQuickAiSearchExecutor({
				linkupApiKey: config.linkupApiKey,
				searchApiKey: config.searchApiKey,
				llmProvider: config.llmProvider,
				model: config.defaultModel,
			}),
		);
	} else if (config.linkupApiKey || config.searchApiKey) {
		registry.register(
			"quick_ai_search",
			quickAiSearchDefinition,
			createQuickAiSearchExecutor({
				linkupApiKey: config.linkupApiKey,
				searchApiKey: config.searchApiKey,
			}),
		);
	}

	registry.register(
		"coworker_text2im",
		coworkerText2ImDefinition,
		createText2ImExecutor(config.imagenApiKey),
	);

	registry.register(
		"create_custom_api_integration",
		createCustomApiIntegrationDefinition,
		createCustomApiIntegrationExecutor,
	);

	registry.register("workspace_tree", workspaceTreeDefinition, workspaceTreeExecutor);

	if (config.slackToken) {
		const slackComms = createSlackToolExecutors(config.slackToken);
		const local = { localOnly: true };
		registry.register(
			"coworker_slack_history",
			coworkerSlackHistoryDefinition,
			slackComms.coworker_slack_history,
			local,
		);
		registry.register(
			"coworker_send_slack_message",
			coworkerSendSlackMessageDefinition,
			slackComms.coworker_send_slack_message,
			local,
		);
		registry.register(
			"coworker_slack_react",
			coworkerSlackReactDefinition,
			slackComms.coworker_slack_react,
			local,
		);
		registry.register(
			"coworker_delete_slack_message",
			coworkerDeleteSlackMessageDefinition,
			slackComms.coworker_delete_slack_message,
			local,
		);
		registry.register(
			"coworker_update_slack_message",
			coworkerUpdateSlackMessageDefinition,
			slackComms.coworker_update_slack_message,
			local,
		);
		// Upload/download need filesystem access — run in Modal (not localOnly)
		// so they can see files created by bash/file_write in the sandbox.
		registry.register(
			"coworker_upload_to_slack",
			coworkerUploadToSlackDefinition,
			slackComms.coworker_upload_to_slack,
		);
		registry.register(
			"coworker_download_from_slack",
			coworkerDownloadFromSlackDefinition,
			slackComms.coworker_download_from_slack,
		);
		const slackAdmin = createSlackAdminExecutors(config.slackToken);
		registry.register(
			"coworker_list_slack_channels",
			coworkerListSlackChannelsDefinition,
			slackAdmin.coworker_list_slack_channels,
			local,
		);
		registry.register(
			"coworker_join_slack_channels",
			coworkerJoinSlackChannelsDefinition,
			slackAdmin.coworker_join_slack_channels,
			local,
		);
		registry.register(
			"coworker_open_slack_conversation",
			coworkerOpenSlackConversationDefinition,
			slackAdmin.coworker_open_slack_conversation,
			local,
		);
		registry.register(
			"coworker_leave_slack_channels",
			coworkerLeaveSlackChannelsDefinition,
			slackAdmin.coworker_leave_slack_channels,
			local,
		);
		registry.register(
			"coworker_list_slack_users",
			coworkerListSlackUsersDefinition,
			slackAdmin.coworker_list_slack_users,
			local,
		);
		registry.register(
			"coworker_invite_slack_user_to_team",
			coworkerInviteSlackUserToTeamDefinition,
			slackAdmin.coworker_invite_slack_user_to_team,
			local,
		);
		registry.register(
			"coworker_get_slack_reactions",
			coworkerGetSlackReactionsDefinition,
			slackAdmin.coworker_get_slack_reactions,
			local,
		);
		registry.register(
			"coworker_report_issue",
			coworkerReportIssueDefinition,
			slackAdmin.coworker_report_issue,
			local,
		);
	}

	if (config.telegramToken) {
		const telegramComms = createTelegramToolExecutors(config.telegramToken);
		const local = { localOnly: true };
		registry.register(
			"coworker_send_telegram_message",
			coworkerSendTelegramMessageDefinition,
			telegramComms.coworker_send_telegram_message,
			local,
		);
		registry.register(
			"coworker_edit_telegram_message",
			coworkerEditTelegramMessageDefinition,
			telegramComms.coworker_edit_telegram_message,
			local,
		);
		registry.register(
			"coworker_delete_telegram_message",
			coworkerDeleteTelegramMessageDefinition,
			telegramComms.coworker_delete_telegram_message,
			local,
		);
	}

	const gitExecutors = createGitExecutors(config.githubToken);
	registry.register("coworker_git", coworkerGitDefinition, gitExecutors.coworker_git);
	registry.register(
		"coworker_github_cli",
		coworkerGithubCliDefinition,
		gitExecutors.coworker_github_cli,
	);

	if (config.browserbaseApiKey) {
		const browserExecutors = createBrowserExecutors(
			config.browserbaseApiKey,
			config.browserbaseProjectId,
		);
		registry.register(
			"browser_create_session",
			browserCreateSessionDefinition,
			browserExecutors.browser_create_session,
		);
		registry.register(
			"browser_download_files",
			browserDownloadFilesDefinition,
			browserExecutors.browser_download_files,
		);
		registry.register(
			"browser_close_session",
			browserCloseSessionDefinition,
			browserExecutors.browser_close_session,
		);
	}

	const docsExecutors = createDocsExecutors(config.context7BaseUrl);
	registry.register(
		"resolve_library_id",
		resolveLibraryIdDefinition,
		docsExecutors.resolve_library_id,
	);
	registry.register(
		"query_library_docs",
		queryLibraryDocsDefinition,
		docsExecutors.query_library_docs,
	);

	return registry;
}

export function registerDbTools(registry: ToolRegistry, prisma: PrismaClient): void {
	const local = { localOnly: true };
	registry.register(
		"read_learnings",
		readLearningsDefinition,
		createReadLearningsExecutor(prisma),
		local,
	);
	registry.register(
		"write_learning",
		writeLearningDefinition,
		createWriteLearningExecutor(prisma),
		local,
	);
	registry.register("read_skill", readSkillDefinition, createReadSkillExecutor(prisma), local);
	registry.register("list_skills", listSkillsDefinition, createListSkillsExecutor(prisma), local);
	registry.register("write_skill", writeSkillDefinition, createWriteSkillExecutor(prisma), local);
}

export {
	listAvailableIntegrationsDefinition,
	listWorkspaceConnectionsDefinition,
	connectIntegrationDefinition,
	disconnectIntegrationDefinition,
	syncWorkspaceConnectionsDefinition,
	createListAvailableIntegrationsExecutor,
	createListWorkspaceConnectionsExecutor,
	createConnectIntegrationExecutor,
	createDisconnectIntegrationExecutor,
	createSyncWorkspaceConnectionsExecutor,
	createIntegrationSyncHandler,
	restoreToolsFromDb,
	convertConfigurableProps,
	actionKeyToToolName,
	extractToolSchemas,
} from "./integrations/index.js";
export type { IntegrationSyncHandler } from "./integrations/index.js";

export type SlackTokenResolver = (workspaceId: string) => string | null;

export function registerDynamicSlackTools(
	registry: ToolRegistry,
	resolveToken: SlackTokenResolver,
): void {
	function wrap(
		factory: (token: string) => Record<string, ToolExecutor>,
		toolName: string,
	): ToolExecutor {
		return async (args, ctx) => {
			const token = resolveToken(ctx.workspaceId);
			if (!token) {
				return {
					output: null,
					durationMs: 0,
					error: `No Slack token available for workspace ${ctx.workspaceId}`,
				};
			}
			return factory(token)[toolName](args, ctx);
		};
	}

	const local = { localOnly: true };

	// Slack comms tools
	registry.register(
		"coworker_slack_history",
		coworkerSlackHistoryDefinition,
		wrap(createSlackToolExecutors, "coworker_slack_history"),
		local,
	);
	registry.register(
		"coworker_send_slack_message",
		coworkerSendSlackMessageDefinition,
		wrap(createSlackToolExecutors, "coworker_send_slack_message"),
		local,
	);
	registry.register(
		"coworker_slack_react",
		coworkerSlackReactDefinition,
		wrap(createSlackToolExecutors, "coworker_slack_react"),
		local,
	);
	registry.register(
		"coworker_delete_slack_message",
		coworkerDeleteSlackMessageDefinition,
		wrap(createSlackToolExecutors, "coworker_delete_slack_message"),
		local,
	);
	registry.register(
		"coworker_update_slack_message",
		coworkerUpdateSlackMessageDefinition,
		wrap(createSlackToolExecutors, "coworker_update_slack_message"),
		local,
	);
	registry.register(
		"coworker_upload_to_slack",
		coworkerUploadToSlackDefinition,
		wrap(createSlackToolExecutors, "coworker_upload_to_slack"),
	);
	registry.register(
		"coworker_download_from_slack",
		coworkerDownloadFromSlackDefinition,
		wrap(createSlackToolExecutors, "coworker_download_from_slack"),
	);

	// Slack admin tools
	registry.register(
		"coworker_list_slack_channels",
		coworkerListSlackChannelsDefinition,
		wrap(createSlackAdminExecutors, "coworker_list_slack_channels"),
		local,
	);
	registry.register(
		"coworker_join_slack_channels",
		coworkerJoinSlackChannelsDefinition,
		wrap(createSlackAdminExecutors, "coworker_join_slack_channels"),
		local,
	);
	registry.register(
		"coworker_open_slack_conversation",
		coworkerOpenSlackConversationDefinition,
		wrap(createSlackAdminExecutors, "coworker_open_slack_conversation"),
		local,
	);
	registry.register(
		"coworker_leave_slack_channels",
		coworkerLeaveSlackChannelsDefinition,
		wrap(createSlackAdminExecutors, "coworker_leave_slack_channels"),
		local,
	);
	registry.register(
		"coworker_list_slack_users",
		coworkerListSlackUsersDefinition,
		wrap(createSlackAdminExecutors, "coworker_list_slack_users"),
		local,
	);
	registry.register(
		"coworker_invite_slack_user_to_team",
		coworkerInviteSlackUserToTeamDefinition,
		wrap(createSlackAdminExecutors, "coworker_invite_slack_user_to_team"),
		local,
	);
	registry.register(
		"coworker_get_slack_reactions",
		coworkerGetSlackReactionsDefinition,
		wrap(createSlackAdminExecutors, "coworker_get_slack_reactions"),
		local,
	);
	registry.register(
		"coworker_report_issue",
		coworkerReportIssueDefinition,
		wrap(createSlackAdminExecutors, "coworker_report_issue"),
		local,
	);
}

export type { ThreadOrchestrationDeps };

export function registerThreadOrchestrationTools(
	registry: ToolRegistry,
	deps: ThreadOrchestrationDeps,
): void {
	registry.register("create_thread", createThreadDefinition, createCreateThreadExecutor(deps));
	registry.register(
		"send_message_to_thread",
		sendMessageToThreadDefinition,
		createSendMessageToThreadExecutor(deps),
	);
	registry.register("wait_for_paths", waitForPathsDefinition, createWaitForPathsExecutor(deps));
	registry.register(
		"list_running_paths",
		listRunningPathsDefinition,
		createListRunningPathsExecutor(deps),
	);
	registry.register("get_path_info", getPathInfoDefinition, createGetPathInfoExecutor(deps));
}

export type { SpacesService };

export function registerSpacesTools(registry: ToolRegistry, service: SpacesService): void {
	const executors = createSpacesToolExecutors(service);
	const local = { localOnly: true };
	for (const definition of spacesToolDefinitions) {
		registry.register(definition.name, definition, executors[definition.name], local);
	}
}
