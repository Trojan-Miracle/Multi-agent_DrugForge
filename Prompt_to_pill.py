import asyncio
import json
import os
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from autogen_ext.models.openai import OpenAIChatCompletionClient

from state import RunState
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.tools.mcp import StdioServerParams, StreamableHttpServerParams, mcp_server_tools

messages_store: list = []
_run: RunState | None = None

class VizHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open("viz.html", "rb") as f:
                self.wfile.write(f.read())
        elif self.path == "/messages":
            data = json.dumps(messages_store, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

    def log_message(self, format, *args):
        pass  # 屏蔽HTTP请求日志

def start_viz_server():
    server = HTTPServer(("localhost", 8765), VizHandler)
    server.serve_forever()

async def broadcast(agent: str, msg_type: str, content: str):
    messages_store.append({
        "id": len(messages_store),
        "agent": agent,
        "type": msg_type,
        "content": content,
    })
    if _run:
        _run.append(agent, msg_type, content)

def _unwrap_json_blocks(s: str) -> str:
    """MCP 返回内容可能是多行 [{"type":"text","text":"..."}]，逐行解包"""
    lines = s.strip().split("\n")
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and '"type"' in line:
            try:
                blocks = json.loads(line)
                if isinstance(blocks, list):
                    texts = [b["text"] for b in blocks
                             if isinstance(b, dict) and b.get("type") == "text" and "text" in b]
                    if texts:
                        results.append("\n".join(texts))
                        continue
            except Exception:
                pass
        results.append(line)
    return "\n".join(results)

def _extract_content(content) -> str:
    if isinstance(content, str):
        return _unwrap_json_blocks(content)
    if not isinstance(content, list):
        return _unwrap_json_blocks(str(content))
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(_unwrap_json_blocks(item))
        elif isinstance(item, dict):
            parts.append(item.get("text") or json.dumps(item, ensure_ascii=False))
        elif hasattr(item, "name") and hasattr(item, "arguments"):
            parts.append(f"[工具调用] {item.name}({item.arguments})")
        elif hasattr(item, "call_id") and hasattr(item, "content"):
            result = _unwrap_json_blocks(str(item.content))
            if len(result) > 500:
                result = result[:500] + "..."
            parts.append(f"[工具结果]\n{result}")
        else:
            text = getattr(item, "text", None) or getattr(item, "content", None)
            parts.append(_unwrap_json_blocks(str(text) if text else repr(item)))
    return "\n".join(parts)

_SKIP_TYPES = {"ToolCallExecutionEvent"}

def _validate_output():
    for msg in reversed(messages_store):
        if msg["agent"] == "planning_agent" and "FINAL" in msg["content"]:
            if len(msg["content"]) > 200:
                print("[验证] 最终报告完整", flush=True)
            else:
                print("[验证] 警告：planning_agent 说了 FINAL 但报告内容过短", flush=True)
            return
    print("[验证] 警告：未检测到 FINAL 报告，流程可能提前终止", flush=True)

async def run_with_viz(team, task):
    async for msg in team.run_stream(task=task):
        msg_type = type(msg).__name__
        if msg_type in _SKIP_TYPES:
            continue
        agent = getattr(msg, "source", "系统")
        raw = getattr(msg, "content", "")
        text = _extract_content(raw)
        if msg_type == "ThoughtEvent" and (not text.strip() or text.strip().startswith("{")):
            continue
        await broadcast(agent, msg_type, text)

DRUGGEN_SYS = (
    '''
    你是药物生成专家。你的任务是根据指定的生物靶点设计新颖的SMILES分子。
    你有一个工具run_druggen，可以根据给定的靶点返回药物SMILES。始终生成7个分子。
    在任何情况下都不要进行性质预测、分子对接或自由文本生成。
    只运行run_druggen工具并返回其结果。如果任务中已存在SMILES，则不要运行该工具。
    '''
)

CHEM_PROPERTIES_SYS = (
    '''
    你是化学性质专家。你的任务是预测并报告分子的化学性质。
    在任何情况下都不要进行临床试验生成、药物生成、ADMET预测或分子对接。
    你有以下工具：
        - select_leads_from_smiles(smiles_list, n)：根据Lipinski规则（MW <=500, logP <=5, HBD <=5, HBA <=10）和Veber规则（RotB <=10, TPSA <=140）筛选n个铅化合物。n是要选择的铅化合物数量。
        - predict_pka_batch：预测SMILES列表的pKa值。
        - logd_acid_batch / logd_base_batch：计算酸或碱在给定pH下的logD值。
        - rdkit_physchem_batch：计算理化描述符（MW、logP、TPSA、HBD、HBA、RotB、QED等）以及pKa和logD（酸/碱）。
        - predict_all_batch：整合以上所有功能，根据酸/碱标志选择相关的logD。
    '''
)

ADMET_PROPERTIES_SYS = (
    '''
    你是ADMET性质专家。你的任务是预测并报告分子的ADMET性质。
    在任何情况下都不要生成临床试验、药物或自由文本。始终先运行chemfm_list_properties获取精确的性质名称。
    选择10个最相关的ADMET性质（例如口服生物利用度、溶解度、清除率、BBB通透性、hERG抑制、肝毒性）。
    你有以下工具：
        - get_drug_name_from_smiles(smiles)：将SMILES列表解析为PubChem CID并返回{"cid","preferred_name","synonyms","iupac_name"}。
                仅当提供了SMILES时使用，检查生成的SMILES是否有可用名称；不要自己编造名称。如果找不到CID，返回工具的错误消息。
        - get_smiles_from_drug_name：从药物名称查找SMILES。不要编造名称，只有当任务中有药物名称时才运行此工具。
        - chemfm_list_properties：列出ChemFM Space支持的所有性质。
        - chemfm_get_description(property_name)：获取某个性质的ChemFM描述。
        - chemfm_predict_single(smiles, property_name)：预测单个SMILES的单个性质，不要传入SMILES列表。
        - chemfm_predict_many(smiles, properties)：预测单个SMILES的多个性质，不要传入SMILES列表。
        - run_docking：执行分子对接并返回对接评分。可用于筛选某靶点的最佳候选药物，评分越低候选药物越好。
                不要在未运行工具的情况下预测对接评分。用于初始命中物筛选和优化后的重新评估。
    '''
)

MOL_OPT_SYS = (
    '''
    你是分子优化专家。你有一个工具：
      - molecule_optimizer(smiles, properties, action)：输入一个SMILES、一个需要优化的性质以及期望的动作（increase增加或decrease减少）。
    不要生成临床试验或药物，不要预测性质或对接评分。不要生成自由文本，只返回工具结果。
    如果ADMET和化学性质尚未预测和评估，则不要运行。
    '''
)

TRIAL_SYS = (
    """
    你是临床试验设计专家。
    你的任务是为药物生成真实的临床试验方案。如果药物由druggen agent生成，使用对接评分最低的药物。
    你不能预测性质、执行对接或生成分子，不要尝试这些任务。
    如果任务中有药物名称，则使用该名称运行。你必须在有SMILES或药物名称的情况下生成试验。
    如果药物名称和药物SMILES都不可用，则不要生成试验。

    首先，按以下格式输出初始试验文本：
- drug: SMILES（如有名称则附上NAME）

CLINICAL TRIAL:
- acronym: 字符串，简短研究名称
- brief_title: 字符串
- official_title: 字符串，描述性试验标题
- study_status: 字符串
- study_start_date: 字符串，ISO格式（例如"2026-03"）
- primary_completion_date: 字符串，ISO格式
- completion_date: 字符串，ISO格式
- condition: 字符串，研究的临床适应症
- study_type: 字符串
- phase: 字符串
- enrollment: 整数
    然后，以整段文本作为trial_text参数调用panacea_extract_components工具。

    最后，使用工具输出构建并输出以下结构化格式的最终报告：
- drug: SMILES（如有名称则附上NAME）

CLINICAL TRIAL:
- acronym: 字符串，简短研究名称
- brief_title: 字符串
- official_title: 字符串，描述性试验标题
- study_status: 字符串（例如"Recruiting"、"Completed"）
- study_start_date: 字符串，ISO格式（例如"2026-03"）
- primary_completion_date: 字符串，ISO格式
- completion_date: 字符串，ISO格式
- condition: 字符串，研究的临床适应症
- study_type: 字符串
- phase: 字符串
- intervention_model: 字符串
- allocation: 字符串
- masking: 字符串
- enrollment: 整数
- arms: [来自工具输出的arms列表]
- intervention_description: 字符串，必须通过SMILES引用该分子
- primary_outcomes: [来自工具输出的outcomes列表]
- secondary_outcomes: [来自工具输出的outcomes列表]
- other_outcomes: []（或从secondary中获取）
- eligibility_criteria: {"inclusion": [来自工具的列表], "exclusion": [来自工具的列表]}
- study_documents: 字符串列表
- brief_summary: 字符串，对试验目的、设计、干预措施和入组资格的简洁描述。

    临床试验阶段规则：
        - 第一阶段（Phase 1）：安全性优先
            从实验室测试过渡到人体试验是医学研究的关键里程碑。
            第一阶段是新疗法首次在人体中测试，以安全性为首要考量。
            辛辛那提大学医学中心指出：
                "I期试验主要关注新药在约20-100名健康志愿者中的安全性和剂量范围。"
            主要特征：
                关注安全性和副作用
                确定最佳给药剂量
                通常涉及健康志愿者
                历时数月完成
        - 第二阶段（Phase 2）：验证有效性
            在确立基本安全参数后，研究者转向评估疗法的有效性。
            第二阶段是科学家开始了解疗法对目标适应症效果的关键步骤。
            主要方面：
                100-300名参与者
                针对特定疾病测试有效性
                持续监测副作用
                通常持续数月至两年
        - 第三阶段（Phase 3）：对比测试
            第三阶段是最全面的评估阶段，研究者将新疗法与当前标准疗法进行比较。
            FDA说明："研究参与者：300至3000名患有该疾病或症状的志愿者。研究时长：1至4年。目的：评估疗效并监测不良反应。"
            主要特征：
                大规模测试（300-3000名参与者）
                与标准疗法对比
                多地点测试
                随机对照组
                持续1-4年
    """
)

PATIENT_MATCHING_SYS = (
    '''
    你是患者匹配专家。你有一个工具：
        - match_patient_trial(xml_path: str, trial_text: str)：该工具将患者与试验进行匹配，
        需要患者摘要XML文件路径和试验摘要文本。在未提供XML患者摘要路径的情况下，不要运行此工具。
    不要编造患者或试验文本。使用trial_generation_agent提供的精确结构化试验文本。
    返回匹配患者数量和匹配患者ID列表。
    '''
)

TRIAL_PRED_SYS = (
    '''
    你是试验预测Agent。根据提供的试验文本预测试验成功概率。
    只返回概率分数（例如0.75）。不要生成试验或药物，不要预测性质或对接评分。不要生成自由文本，只返回工具结果。
    '''
)

SELECTOR_PROMPT = (
    '''
    选择一个agent来执行任务。

    {roles}

    当前对话上下文：
    {history}

    阅读以上对话，然后从{participants}中选择一个agent执行下一个任务。
    严格遵循planning_agent的计划，包括阶段性工作流程（药物发现→临床前→临床）和优化循环。
    确保planning_agent在其他agent开始工作前已分配任务。
    只选择planning_agent计划中列出的agent。
    在获得具有良好ADMET、化学性质、对接评分的最终优化铅化合物并选定单一铅化合物之前，不要选择trial_generation_agent。
    如果ADMET性质尚未以数值形式预测，不要选择molecule_optimization_agent。
    优化后不要跳过对接或性质重新评估步骤。
    如果某个agent在一次重试后仍未产生输出，记录失败并选择planning_agent处理错误并继续。
    一旦planning_agent输出最终报告，返回"FINAL"，不再选择其他agent并终止流程。
    只选择一个agent。不要选择不在planning_agent计划中的agent。
    当某个agent被调用一次并产生输出后，不要多次调用它，阅读对话历史并使用该响应。
    '''
)

async def main():
    sys.stdout.reconfigure(encoding="utf-8")

    _DS_INFO = {
        "vision": False, "function_calling": True,
        "json_output": True, "family": "unknown", "structured_output": False,
    }
    def ds(thinking: bool = False, pro: bool = False) -> OpenAIChatCompletionClient:
        model = "deepseek-v4-pro" if pro else "deepseek-v4-flash"
        kw = {"extra_body": {"thinking": {"type": "enabled"}}} if thinking else {}
        return OpenAIChatCompletionClient(
            model=model,
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            model_info=_DS_INFO,
            **kw,
        )

    druggen_tools = await mcp_server_tools(StdioServerParams(
            command="python", args=["druggen_mcp_server.py"], read_timeout_seconds=600))
    docking_tools = await mcp_server_tools(StdioServerParams(
            command="python", args=["docking_mcp_server.py"], read_timeout_seconds=1800))
    chemical_properties_tools = await mcp_server_tools(StdioServerParams(
            command="python", args=["chemical_properties_mcp_server.py"], read_timeout_seconds=1500))
    admet_properties_tools = await mcp_server_tools(StdioServerParams(
            command="python", args=["admet_prediction_mcp_server.py"], read_timeout_seconds=1500))
    name_tools = await mcp_server_tools(
            StdioServerParams(command="python", args=["name2smiles_mcp_server.py"], read_timeout_seconds=150))
    optimization_tools = await mcp_server_tools(
            StdioServerParams(command="python", args=["mol_opt_mcp_server.py"], read_timeout_seconds=1500))
    panacea_patient_tools = await mcp_server_tools(
            StdioServerParams(command="python", args=["patient_matching_mcp_server.py"], read_timeout_seconds=3000))
    panacea_trial_tools = await mcp_server_tools(
            StdioServerParams(command="python", args=["trialgen_mcp_server.py"], read_timeout_seconds=3000))
    meditab_tools = await mcp_server_tools(
            StdioServerParams(command="python", args=["trialpred_mcp_server.py"], read_timeout_seconds=3000))

    planning_agent = AssistantAgent(
        name="planning_agent",
        description="负责任务规划的agent，收到新任务时应最先介入。",
        model_client=ds(thinking=True),
        system_message="""
        你是规划agent。
        你的职责是将复杂任务拆解为更小的可管理子任务。
        你的团队成员包括：
            1. druggen_agent：仅根据给定的UniProt ID生成药物SMILES，如果任务中未提供药物则使用它。
            2. chemical_agent — 化学性质专家
                 职责范围：预测/报告化学性质并返回铅化合物（不生成试验/药物）。
                 可用工具：
                   - select_leads_from_smiles(smiles_list)
                   - predict_pka_batch(smiles_list)
                   - logd_acid_batch(smiles_list, pH=7.4)
                   - logd_base_batch(smiles_list, pH=7.4)
                   - rdkit_physchem_batch(smiles_list, pH=7.4)
                   - predict_all_batch(smiles_list, is_acid=None|bool, is_base=None|bool, pH=7.4)
            3. admet_properties_agent — ADMET性质专家
                 职责范围：预测/报告ADMET性质和对接；可解析名称/SMILES（不生成试验/药物）。
                 选择主要铅化合物时只选一个（最佳的）。
                 可用工具：
                   - get_drug_name_from_smiles(smiles_list)
                   - get_smiles_from_drug_name(drug_name)
                   - chemfm_list_properties()
                   - chemfm_get_description(property_name)
                   - chemfm_predict_single(smiles, property_name)
                   - chemfm_predict_many(smiles, properties_list)
                   - run_docking(target, smiles_list)
                 始终先规划chemfm_list_properties工具，以获取运行其他ADMET工具所需的精确性质名称。
            4. molecule_optimization_agent：针对需要增加/减少的性质优化铅化合物。
               必须传入ADMET或化学性质，不能传入对接相关参数。
            5. trial_generation_agent：为给定的铅化合物生成临床试验，在需要生成试验时使用。
               除非它是唯一需要的agent，否则始终在admet agent和chem agent之后最后调用。
               始终以如下结构化格式返回：
               - drug: SMILES和NAME

CLINICAL TRIAL:
- acronym: 字符串，简短研究名称
- brief_title: 字符串
- official_title: 字符串，描述性试验标题
- study_status: 字符串（例如"Recruiting"、"Completed"）
- study_start_date: 字符串，ISO格式（例如"2026-03"）
- primary_completion_date: 字符串，ISO格式
- completion_date: 字符串，ISO格式
- condition: 字符串，研究的临床适应症
- study_type: 字符串
- phase: 字符串
- intervention_model: 字符串
- allocation: 字符串
- masking: 字符串
- enrollment: 整数
- arms: 使用panacea_extract_components工具的输出，以药物SMILES、brief_title、official_title、phase和condition作为输入
- intervention_description: 字符串，必须通过SMILES引用该分子
- primary_outcomes: 使用panacea_extract_components工具的输出
- secondary_outcomes: 使用panacea_extract_components工具的输出
- other_outcomes: 使用panacea_extract_components工具的输出
- eligibility_criteria: 使用panacea_extract_components工具的输出
- study_documents: 字符串列表
- brief_summary: 字符串，对试验目的、设计、干预措施和入组资格的简洁描述。
            6. patient_matching_agent：将患者与试验匹配。未提供XML患者摘要路径时不要规划此agent。
               不要给它编造的试验摘要文本，严格使用trial_generation_agent生成的试验摘要文本。
            7. trial_prediction_agent：根据trial_generation_agent的结构化输出预测试验成功概率。

        规则：
            严格遵循以下药物开发任务工作流程：
            - 药物发现阶段：
              - 命中物生成：druggen_agent生成>10个SMILES。
              - 对接：admet_properties_agent使用run_docking对命中物评分和筛选（保留评分最低的）。
              - 铅化合物鉴定：chemical_agent进行理化性质计算和select_leads_from_smiles（应用Lipinski/Veber过滤）；admet_properties_agent评估10个相关ADMET性质。
              - 选择单个最佳铅化合物（对接评分最低+通过过滤器+最佳ADMET，例如生物利用度>0.5，低hERG/肝毒性）。
            - 铅化合物优化阶段：
              - 对最佳铅化合物使用molecule_optimization_agent，针对薄弱性质（例如生物利用度<0.5时提高，降低毒性风险）。
              - 重新评估：admet_properties_agent（对接+ADMET），chemical_agent（理化性质/Lipinski/Veber）。
              - 循环：如果不满意（未通过Lipinski/Veber，生物利用度<0.5，高毒性风险，对接评分差于初始铅化合物），规划另一次优化针对问题。最多3次迭代；如果仍不满意，选择现有最佳并继续。
            - 临床前阶段：用admet_properties_agent对优化后的铅化合物进行最终ADMET重新评估。
            - 临床阶段：对最终铅化合物使用trial_generation_agent；如果有XML路径，使用patient_matching_agent；然后使用trial_prediction_agent。
            - 当被要求生成药物时，只生成候选药物，不要生成试验，并用get_drug_name_from_smiles检查生成的SMILES是否有可用名称。
            - run_docking必须在命中物生成后和每次优化后运行。
            - 最终报告格式如下：
                药物发现性质：
                - 命中物：[初始SMILES列表]
                - 铅化合物：[带性质的已选铅化合物列表]
                - 优化铅化合物：带优化性质的SMILES
                - 化学性质：[来自chem agent的字典或列表]
                - ADMET性质：[来自admet agent的字典或列表]
                - 对接评分：[评分]

                临床试验报告：
                [来自trial agent的完整结构化试验]

                患者匹配：[结果（如适用）]

                试验成功概率：[概率]

                总结：[整体总结]
            - 不要编造agent的响应。如果对话历史中没有某agent的响应，规划调用该agent。
            - 始终检查优化后新分子的ADMET和化学性质，如果不好则最多运行3次优化。
            - ADMET预测使用ADMET性质列表中最相关的10个性质（例如生物利用度、溶解度、清除率、BBB通透性、hERG抑制、肝毒性）。
            - 如果任何agent或工具未能产生输出，记录失败并使用最佳现有数据继续，选择最佳铅化合物或优化分子继续。
            - 所有任务完成后或无法进一步推进时，以指定格式输出最终报告。
        不要编造新的化学或ADMET测试，只使用可用的性质，并根据它们选择后续步骤和生成报告。
        如果需要额外测试，只需建议，但说明没有工具执行，然后继续使用现有数据。不要编造对接评分。
        规划试验生成时，必须有铅化合物，或任务中必须指定化合物。
        不要自己编造或选择SMILES，也不要优先选择有名称的SMILES。不要以任何方式规划get_drug_name_from_smiles来过滤SMILES列表，除非明确要求。
        你只负责规划和委派任务，不自己执行。不需要在计划中包含所有agent。
        在规划trial_generation_agent之前，始终说明选择该药物的原因和理由。
        分配任务时使用以下格式：
        1. <agent>：<任务>

        所有任务完成、所需工具已执行，或任何步骤失败且无法进一步推进时，输出最终报告并以FINAL终止流程。
        除终止流程外，在任何计划说明中都不要出现"FINAL"或其任何变体。
        """
    )

    druggen_agent = AssistantAgent(
        name="druggen_agent",
        description="根据UniProt ID生成候选药物SMILES，仅在需要全新分子生成时调用。",
        model_client=ds(),
        tools=list(druggen_tools),
        system_message=DRUGGEN_SYS,
        reflect_on_tool_use=True,
    )
    chemical_agent = AssistantAgent(
        name="chemical_agent",
        description="预测分子理化性质并按Lipinski/Veber规则筛选铅化合物，不做ADMET或对接。",
        model_client=ds(),
        tools=list(chemical_properties_tools),
        system_message=CHEM_PROPERTIES_SYS,
        reflect_on_tool_use=True,
    )
    admet_properties_agent = AssistantAgent(
        name="admet_properties_agent",
        description="预测ADMET性质和分子对接评分，用于铅化合物筛选和优化后重评估。",
        model_client=ds(),
        tools=list(admet_properties_tools) + list(docking_tools) + list(name_tools),
        system_message=ADMET_PROPERTIES_SYS,
        reflect_on_tool_use=True,
    )
    molecule_optimization_agent = AssistantAgent(
        name="molecule_optimization_agent",
        description="优化铅化合物的特定性质，须在ADMET和化学性质已评估后调用，最多3次。",
        model_client=ds(thinking=True),
        tools=list(optimization_tools),
        system_message=MOL_OPT_SYS,
        reflect_on_tool_use=True,
    )
    trial_generator_agent = AssistantAgent(
        name="trial_generator_agent",
        description="为最终优化铅化合物生成临床试验方案，必须是药物发现和优化阶段完成后最后调用。",
        model_client=ds(thinking=True),
        tools=list(panacea_trial_tools),
        system_message=TRIAL_SYS,
        reflect_on_tool_use=True,
    )
    patient_matching_agent = AssistantAgent(
        name="patient_matching_agent",
        description="将XML患者数据与临床试验标准匹配，需提供XML路径和试验文本才能调用。",
        model_client=ds(),
        tools=list(panacea_patient_tools),
        system_message=PATIENT_MATCHING_SYS,
        reflect_on_tool_use=True,
    )
    trial_prediction_agent = AssistantAgent(
        name="trial_prediction_agent",
        description="根据试验方案预测临床试验成功概率，返回0到1之间的概率值。",
        model_client=ds(),
        tools=list(meditab_tools),
        system_message=TRIAL_PRED_SYS,
        reflect_on_tool_use=True,
    )

    termination = TextMentionTermination("FINAL")

    task = input("请输入你的任务：").strip() or "模拟DPP4(P27487)的药物开发"

    global _run
    _run = RunState(task=task)
    run: RunState = _run
    print(f"Run ID: {run.run_id}", flush=True)

    participants = [
        planning_agent, druggen_agent, chemical_agent, admet_properties_agent,
        molecule_optimization_agent, trial_generator_agent,
        patient_matching_agent, trial_prediction_agent
    ]
    team = SelectorGroupChat(
        participants=participants,
        model_client=ds(),
        selector_prompt=SELECTOR_PROMPT,
        termination_condition=termination,
        max_turns=30,
    )

    threading.Thread(target=start_viz_server, daemon=True).start()
    print("\n可视化面板：http://localhost:8765", flush=True)
    print("按回车开始运行...", flush=True)
    input()

    try:
        await run_with_viz(team, task)
        _validate_output()
        run.done()
    except Exception as e:
        print(f"运行出错：{e}", flush=True)
        run.failed(str(e))
    finally:
        print(f"\n运行完成  run_id={run.run_id}", flush=True)
        print("服务器保持运行，可刷新浏览器查看结果", flush=True)
        while True:
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
