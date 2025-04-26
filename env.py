from utils import (load_config,
                   generate_levels,
                   distribute_evenly,
                   distribute_elements,
                   inverse_weight_normalized,
                   gini,
                   RunningMeanStd,
                   Logger
                   )
import numpy as np


class EconomicEnv:
    def __init__(self):
        self.config = load_config()
        self.logger = Logger()
        self.device = self.config['device']
        self.worker_obs_dim = self.config["size"]["observation"]["worker"]
        self.firm_obs_dim = self.config["size"]["observation"]["firm"]
        self.government_obs_dim = self.config["size"]["observation"]["government"]
        # ============================一些固定参数==========================
        self.num_worker_agents = self.config['num_worker_agents']
        self.num_firm_agents = self.config['num_firm_agents']
        self.investment_rate = self.config['constants']['investment_rate']
        self.depreciation_rate = self.config['constants']['depreciation_rate']
        self.switch_job_penalty = self.config['constants']['switch_job_penalty']
        self.swf_eq_param = self.config['constants']['swf_eq_param']
        self.storage_dpr = self.config['constants']['storage_dpr']
        # ============================一些范围参数==========================
        self.quote_range = np.array(self.config['constants']['quote_range'])
        self.labor_range = np.array(self.config['constants']['labor_range'])
        self.wage_range = np.array(self.config['constants']['wage_range'])
        self.consumption_range = np.array(self.config['constants']['consumption_range'])
        self.tax_rate_range = np.array(self.config['constants']['tax_rate_range'])
        self.interest_rate = self.config['constants']['interest_rate']

        # ====================== 用于标准化输入状态 ==========================
        # # 用于标准化输入状态
        # self.rms_worker = RunningMeanStd(shape=(self.worker_obs_dim,))
        # self.rms_firm = RunningMeanStd(shape=(self.firm_obs_dim,))
        # self.rms_government = RunningMeanStd(shape=(self.government_obs_dim,))

        # # 用于标准化奖励
        # self.rms_worker_reward = RunningMeanStd(shape=(self.num_worker_agents,))
        # self.rms_firm_reward = RunningMeanStd(shape=(self.num_firm_agents,))
        # self.rms_government_reward = RunningMeanStd(shape=(1,))

    def reset(self):
        self.episode_start = True
        # ========================== 劳动者相关 ==========================
        # 劳动者资产
        self.worker_asset = np.full((self.num_worker_agents, ),
                                    self.config['initialize']['worker_asset'], dtype=np.float32)
        # 劳动者技能禀赋
        self.worker_levels = generate_levels(self.num_worker_agents)
        # 劳动者劳动厌恶系数
        self.worker_labor_aversion = distribute_elements(
            self.config['constants']['labor_aversion_range'], self.num_worker_agents)

        # 初始化每个劳动者所属的企业
        self.worker_in_firm = np.array(distribute_evenly(
            self.num_worker_agents, self.num_firm_agents), dtype=np.int32)
        # 劳动者劳动量
        self.worker_labor = np.zeros((self.num_worker_agents,), dtype=np.float32)
        # 劳动者累计消费量(求和各个公司)
        self.worker_total_consumption = np.zeros((self.num_worker_agents,), dtype=np.float32)
        # 劳动者在各企业消费量
        self.worker_consumption = np.zeros((self.num_worker_agents, self.num_firm_agents), dtype=np.float32)
        # 劳动者在上家企业累计工作时长
        # 如果跳槽则清空，若不跳槽继续累计
        self.worker_firm_len = np.zeros((self.num_worker_agents,), dtype=np.float32)
        # 各劳动者资产变化
        self.worker_asset_change = np.zeros((self.num_worker_agents,), dtype=np.float32)
        # 各劳动者消费额
        self.worker_cost = np.zeros((self.num_worker_agents,), dtype=np.float32)
        # ========================== 企业相关 ==========================
        # 企业资产
        self.firm_asset = np.full((self.num_firm_agents,),
                                  self.config['initialize']['firm_asset'], dtype=np.float32)
        # 企业资本
        self.firm_capital = np.full((self.num_firm_agents,),
                                    self.config['initialize']['firm_capital'], dtype=np.float32)
        # 企业资本弹性
        self.firm_capital_elasticity = distribute_elements(
            self.config['constants']['captial_elasticity'], self.num_firm_agents)
        # 各企业报价
        self.firm_quote = np.full((self.num_firm_agents,),
                                  self.config['initialize']['quote'])
        # 各企业工资水平
        self.firm_wage = np.full((self.num_firm_agents,),
                                 self.config['initialize']['wage'])
        # 各企业销售额
        self.firm_sales = np.zeros((self.num_firm_agents,), dtype=np.float32)

        # 各企业所拥有的劳动量
        self.firm_labor = np.zeros((self.num_firm_agents,), dtype=np.float32)
        # 各企业生产量
        self.firm_production = np.zeros((self.num_firm_agents,), dtype=np.float32)
        # 各企业资产变化
        self.firm_asset_change = np.zeros((self.num_firm_agents,), dtype=np.float32)
        # 各企业在自身上的消费者总需求量
        self.firm_total_demand = np.zeros((self.num_firm_agents,), dtype=np.float32)
        # 各企业库存
        self.firm_inventory = np.zeros((self.num_firm_agents,), dtype=np.float32)
        # ========================== 政府相关 ==========================
        # 政府税率
        self.tax_rate = self.config['initialize']['tax_rate']
        # 上一期总转移支付（税收）
        self.total_transfer = 0

        # ========================== 其他参数 ==========================

    def compute_firm_labor(self):
        """根据劳动者的劳动量，计算每个企业的总劳动量
        例如劳动量向量是[100,200,300,400,100]，
        对应工作企业序号向量是[0,1,1,2,0],
        输出为和企业数量维度一致的向量，代表在每家企业的总劳动量
        [200,500,400]
        """
        return np.bincount(self.worker_in_firm, weights=self.worker_labor, minlength=self.num_firm_agents)

    def judge_switch_firm(self, next_worker_in_firm):
        """判断劳动者是否跳槽，若跳槽则对应向量位置为1"""
        return (self.worker_in_firm != next_worker_in_firm).astype(int)

    def construct_worker_obs(self) -> np.ndarray:
        """构造劳动者的部分观测
        一般观测的维度是(num_agents,state_dim)
        - 每个企业的价格
        - 每个企业的工资
        - 上期每个企业的产量
        - 政府税率
        - 自身资产 
        - 在目前公司工作时长
        - 技能系数
        - 劳动厌恶
        - 目前工作公司
        - 上期资产变化
        - 上期消费量（总和）
        - 上期劳动量

        输出维度为(num_worker_agents,worker_obs_dim)
        其中worker_obs_dim=config["size"]["observation"]["worker"]
        """
        worker_obs = np.concatenate([
            self.vector2obs(self.firm_quote, self.num_worker_agents),
            self.vector2obs(self.firm_wage, self.num_worker_agents),
            self.vector2obs(self.firm_production, self.num_worker_agents),
            np.full((self.num_worker_agents, 1), self.tax_rate),
            self.worker_asset[:, np.newaxis],
            self.worker_firm_len[:, np.newaxis],
            self.worker_levels[:, np.newaxis],
            self.worker_labor_aversion[:, np.newaxis],
            self.worker_in_firm[:, np.newaxis],
            self.worker_asset_change[:, np.newaxis],
            self.worker_total_consumption[:, np.newaxis],
            self.worker_labor[:, np.newaxis],
        ], axis=-1).astype(np.float32)

        # # normalization
        # self.rms_worker.update(worker_obs)
        # worker_obs = ((worker_obs-self.rms_worker.mean)/np.sqrt(self.rms_worker.var+1e-5)).astype(np.float32)
        return worker_obs

    def construct_firm_obs(self) -> np.ndarray:
        """构造企业的部分观测
        - 本期每个企业的价格
        - 本期每个企业的工资
        - 上期每个企业的产量
        - 政府税率
        - 自身资产
        - 自身资本
        - 上期资产变动
        - 技术水平参数
        - 上期生产量
        - 上期在自己身上的总需求量
        - 上期库存
        """
        firm_obs = np.concatenate([
            self.vector2obs(self.firm_quote, self.num_firm_agents),
            self.vector2obs(self.firm_wage, self.num_firm_agents),
            self.vector2obs(self.firm_production, self.num_firm_agents),
            np.full((self.num_firm_agents, 1), self.tax_rate),
            self.firm_asset[:, np.newaxis],
            self.firm_capital[:, np.newaxis],
            self.firm_asset_change[:, np.newaxis],
            self.firm_capital_elasticity[:, np.newaxis],
            self.firm_production[:, np.newaxis],
            self.firm_total_demand[:, np.newaxis],
            self.firm_inventory[:, np.newaxis],
        ], axis=-1).astype(np.float32)

        # # normalization
        # self.rms_firm.update(firm_obs)
        # firm_obs = ((firm_obs-self.rms_firm.mean)/np.sqrt(self.rms_firm.var+1e-5)).astype(np.float32)
        return firm_obs

    def construct_government_obs(self) -> np.ndarray:
        """构造政府观测
        - 各个劳动者资产
        - 各个企业资产
        - 本期税率
        - 本期转移支付（总税收）
        - 本期各个劳动者的资产变化
        - 本期各企业的资产变化
        只有一个政府，输出一般是(1,government_obs_dim)
        """
        government_obs = np.concatenate([
            self.worker_asset,
            self.firm_asset,
            np.array([self.tax_rate]),
            np.array([self.total_transfer]),
            self.worker_asset_change,
            self.firm_asset_change,
        ])[np.newaxis, :].astype(np.float32)

        # # normalization
        # self.rms_government.update(government_obs)
        # government_obs = ((government_obs-self.rms_government.mean) /
        #                   np.sqrt(self.rms_government.var+1e-5)).astype(np.float32)
        return government_obs

    def worker_settlement(self, worker_action: np.ndarray) -> None:
        """劳动者执行后结算：
        - 各个公司的劳动总量
        - 存储劳动者的报价报量，等待市场结算
        - 劳动者跳槽指示更新
          其中action的顺序为：消费，劳动，报价，工作企业


        """
        # 每个工人在每家公司的消费量(num_agent,num_firm)，一行是[c_1,c_2,c_3,c_4,c_5]
        self.worker_consumption = self.consumption_range[worker_action[:, :-2]]

        self.logger.info(f"调整前每家公司消费量:\n {self.worker_consumption}\n")
        self.adjust_consumption()
        self.logger.info(f"调整后每家公司消费量:\n {self.worker_consumption}\n")

        self.worker_labor = self.labor_range[worker_action[:, -2]]
        self.logger.info(f"劳动量:\n{self.worker_labor}")
        if self.episode_start:
            next_worker_in_firm = self.worker_in_firm
        else:
            next_worker_in_firm = worker_action[:, -1]
        self.worker_switch_firm = self.judge_switch_firm(next_worker_in_firm=next_worker_in_firm)
        # 更新劳动者所属企业
        self.worker_in_firm = next_worker_in_firm
        self.logger.info(f"劳动者所属企业:\n{self.worker_in_firm}")
        self.logger.info(f"劳动者跳槽:\n{self.worker_switch_firm}")
        # 各个公司的劳动总量
        self.firm_labor = self.compute_firm_labor()

    def adjust_consumption(self):
        """根据劳动者的预算，调整消费量保证不超出预算

        若超出预算则进行：n'_i = n_i \times \frac{B}{\sum_{i=1}^5 n_i p_i}缩放
        """
        # 计算每个工人的总支出
        worker_cost = (self.worker_consumption*self.firm_quote).sum(axis=-1)
        scale = np.minimum(self.worker_asset/worker_cost, 1.0)
        # 对于超出预算的工人消费量进行缩放，即对每一行乘以对应的缩放系数
        self.worker_consumption = np.floor(self.worker_consumption*scale[:, np.newaxis])

    def firm_settlement(self, firm_action: np.ndarray) -> None:
        """企业执行动作后结算
        - 计算企业生产量（根据该企业的劳动量，资本量，资本弹性）
        - 市场清算：根据劳动者的报价报量和企业报价和生产量进行匹配：
            - 计算劳动者消费额p_{i,t}*C_{i,t}
            - 计算企业销售额p_{j,t}*C_{j,t}，销量可能来自多个劳动者
        - 工资结算：企业根据劳动量和智能体水平等，发放工资（具体是一个向量和劳动者维度一致）
        """

        self.firm_quote = self.quote_range[firm_action[:, 0]]
        self.logger.info(f"企业报价:\n{self.firm_quote}")
        self.firm_wage = self.wage_range[firm_action[:, 1]]
        self.logger.info(f"企业工资:\n{self.firm_wage}")
        # 计算企业生产量，根据生产函数
        self.logger.info(f"企业拥有劳动总量:\n{self.firm_labor}")
        self.firm_production = np.floor((self.firm_capital**self.firm_capital_elasticity) *
                                        (self.firm_labor**(1-self.firm_capital_elasticity)))
        self.logger.info(f"企业上期库存:\n{self.firm_inventory}")
        # 更新当前库存
        self.firm_inventory += self.firm_production
        self.adjust_overdemand()
        # 购买完毕 更新库存
        self.firm_inventory -= self.firm_total_demand
        # 库存损耗
        self.firm_inventory = np.floor(self.firm_inventory*(1-self.storage_dpr))
        self.logger.info(f"企业生产量:\n{self.firm_production}")
        self.logger.info(f"企业总需求量:\n{self.firm_total_demand}")
        self.logger.info(f"企业当前库存:\n{self.firm_inventory}")
        # 计算企业销售额
        self.firm_sales = (self.firm_quote*self.worker_consumption).sum(axis=0)
        self.logger.info(f"企业销售额:\n{self.firm_sales}")
        # 计算工人消费额
        self.worker_cost = (self.firm_quote*self.worker_consumption).sum(axis=1)
        self.logger.info(f"工人消费额:\n{self.worker_cost}")
        # 工资结算，对应每个劳动者获得的税前工资
        self.pre_tax_wages = self.worker_labor*self.worker_levels*self.firm_wage[self.worker_in_firm]
        self.logger.info(f"劳动者税前工资:\n{self.pre_tax_wages}")
        # 计算企业支付出的工资
        self.firm_wage_cost = np.bincount(
            self.worker_in_firm, weights=self.pre_tax_wages, minlength=self.num_firm_agents)
        self.logger.info(f"企业支付出的工资:\n{self.firm_wage_cost}")

    def adjust_overdemand(self):
        """根据企业库存，调整消费者消费量，避免超出企业供应"""
        # 计算每个企业的总需求量
        total_firm_demand = self.worker_consumption.sum(axis=0)
        scale = np.minimum(self.firm_inventory/total_firm_demand, 1.0)
        # 对每一列，乘以缩放系数。一列代表一个公司，列求和是在这家公司总消费量
        self.worker_consumption = np.floor(self.worker_consumption*scale)
        # 更新总需求量
        self.firm_total_demand = self.worker_consumption.sum(axis=0)
        # 更新每个工人总消费量
        self.worker_total_consumption = self.worker_consumption.sum(axis=1)

    def government_settlement(self, government_action: np.ndarray) -> None:
        """政府执行动作后结算
        - 计算企业效用，即奖励（就是税前利润）
        - 更新企业资产（计算税前利润）
        - 企业和劳动者征税
        - 计算转移支付（与资产成反比）
        - 更新劳动者资产（上一期资产+转移支付+税前工资-税收-消费额）*（1+利率）
        - 计算劳动者效用，即奖励（根据消费、劳动、跳槽，资产变化）
        - 计算社会效率
        - 计算社会公平性（根据基尼系数）
        - 计算政府奖励
        - 企业资本投入和折旧
        """
        self.tax_rate = self.tax_rate_range[government_action[:, 0]].item()
        # ===========================企业资产更新=======================
        # 企业税前利润，也是效用和奖励
        self.firm_reward = self.firm_sales-self.firm_wage_cost
        new_firm_asset = self.firm_asset+(1-self.tax_rate)*self.firm_reward
        capital_investment = new_firm_asset*self.investment_rate
        capital_investment = np.clip(capital_investment, 0.0, np.inf)
        new_firm_asset = new_firm_asset-capital_investment
        self.firm_asset_change = new_firm_asset-self.firm_asset

        self.logger.info(f"企业旧资产:\n{self.firm_asset}")
        self.logger.info(f"企业新资产:\n{new_firm_asset}")
        self.logger.info(f"企业资产变化:\n{self.firm_asset_change}")
        self.firm_asset = new_firm_asset

        self.firm_capital = self.firm_capital*(1-self.depreciation_rate)+capital_investment
        # 征税
        worker_tax = (self.pre_tax_wages*self.tax_rate).sum()
        firm_tax = (self.firm_reward*self.tax_rate).sum()
        self.total_transfer = worker_tax+firm_tax
        self.logger.info(f"劳动者税:{worker_tax}\t企业税:{firm_tax}")

        # ===========================转移支付==========================
        # 转移支付
        weights = inverse_weight_normalized(self.worker_asset)
        transfer_to_worker = self.total_transfer*weights
        self.logger.info(f"转移支付:\n{transfer_to_worker}")
        # ==========================资产更新==========================
        # 更新劳动者资产
        new_asset = (self.worker_asset*(1+self.interest_rate)+self.pre_tax_wages*(1-self.tax_rate) +
                     transfer_to_worker-self.worker_cost)
        self.worker_asset_change = new_asset-self.worker_asset
        self.logger.info(f"劳动者旧资产:\n{self.worker_asset}")
        self.logger.info(f"劳动者新资产:\n{new_asset}")
        self.logger.info(f"劳动者资产变化:\n{self.worker_asset_change}")
        self.worker_asset = new_asset
        # ==========================效用计算==========================
        # 计算劳动者效用（相对风险厌恶为0.33），也是奖励
        self.worker_utility = (self.worker_total_consumption**0.67)/0.67-self.worker_labor_aversion * \
            self.worker_labor-self.switch_job_penalty*self.worker_switch_firm*self.worker_firm_len

        self.worker_firm_len = (self.worker_firm_len+self.worker_labor)*(1-self.worker_switch_firm)
        self.social_efficiency = self.worker_utility.sum()
        self.equality = 1-(self.num_worker_agents)/(self.num_worker_agents-1)*gini(self.pre_tax_wages)
        self.government_reward = ((self.equality)**self.swf_eq_param)*(self.social_efficiency**(1-self.swf_eq_param))
        self.logger.info(f"劳动者效用:\n{self.worker_utility}")
        self.logger.info(f"企业效用:\n{self.firm_reward}")
        self.logger.info(f"社会效率:\n{self.social_efficiency}\t社会公平性:\n{self.equality}\t政府奖励:\n{self.government_reward}")

        # 奖励标准化
        # self.rms_worker_reward.update(self.worker_utility)
        # self.rms_firm_reward.update(self.firm_reward)
        # self.rms_government_reward.update(self.government_reward)

        # self.worker_utility = ((self.worker_utility-self.rms_worker_reward.mean) /
        #                        np.sqrt(self.rms_worker_reward.var+1e-5)).astype(np.float32)
        # self.firm_reward = ((self.firm_reward-self.rms_firm_reward.mean) /
        #                     np.sqrt(self.rms_firm_reward.var+1e-5)).astype(np.float32)
        # self.government_reward = ((self.government_reward-self.rms_government_reward.mean) /
        #                           np.sqrt(self.rms_government_reward.var+1e-5)).astype(np.float32)

    def scalar_repeat(self, scalar: float, n: int) -> np.ndarray:
        "将标量扩展为形状为 (n, ) 的向量。"
        return np.full((n, ), scalar)

    def vector2obs(self, vector: np.ndarray, K: int) -> np.ndarray:
        """根据指定的agent数目K，把一个一维向量(n,)扩展成(K,n)"""
        return np.tile(vector, (K, 1))

    def market_clearing(self):
        pass
