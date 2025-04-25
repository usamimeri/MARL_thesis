from utils import (load_config,
                   generate_levels,
                   distribute_evenly,
                   distribute_elements,
                   inverse_weight_normalized,
                   gini,
                   sigmoid)
import numpy as np
from utils import RunningMeanStd
import warnings


class EconomicEnv:
    def __init__(self):
        self.config = load_config()
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
        # ============================一些范围参数==========================
        self.quote_range = np.array(self.config['constants']['quote_range'])
        self.labor_range = np.array(self.config['constants']['labor_range'])
        self.wage_range = np.array(self.config['constants']['wage_range'])
        self.consumption_range = np.array(self.config['constants']['consumption_range'])
        self.tax_rate_range = np.array(self.config['constants']['tax_rate_range'])

        # ====================== 用于标准化输入状态 ==========================
        # 用于标准化输入状态
        self.rms_worker = RunningMeanStd(shape=(self.worker_obs_dim,))
        self.rms_firm = RunningMeanStd(shape=(self.firm_obs_dim,))
        self.rms_government = RunningMeanStd(shape=(self.government_obs_dim,))

        # 用于标准化奖励
        self.rms_worker_reward = RunningMeanStd(shape=(self.num_worker_agents,))
        self.rms_firm_reward = RunningMeanStd(shape=(self.num_firm_agents,))
        self.rms_government_reward = RunningMeanStd(shape=(1,))

    def reset(self):
        # ========================== 劳动者相关 ==========================
        # 劳动者资产
        self.worker_asset = np.full((self.num_worker_agents, ),
                                    self.config['initialize']['worker_asset'], dtype=np.float32)
        # 劳动者技能禀赋
        self.worker_levels = generate_levels(self.num_worker_agents)
        # 劳动者劳动厌恶系数
        self.worker_labor_aversion = distribute_elements(
            self.config['constants']['labor_aversion_range'], self.num_worker_agents)
        # 劳动者报价
        self.worker_quote = np.full((self.num_worker_agents,),
                                    self.config['initialize']['quote'], dtype=np.float32)
        # 初始化每个劳动者所属的企业
        self.worker_in_firm = np.array(distribute_evenly(
            self.num_worker_agents, self.num_firm_agents), dtype=np.int32)
        # 劳动者独热编码
        self.worker_one_hot = np.eye(self.num_worker_agents, dtype=np.float32)
        # 劳动者劳动量
        self.worker_labor = np.zeros((self.num_worker_agents,), dtype=np.float32)
        # 劳动者消费量
        self.worker_consumption = np.zeros((self.num_worker_agents,), dtype=np.float32)
        # 劳动者在上家企业累计工作时长
        # 如果跳槽则清空，若不跳槽继续累计
        self.worker_firm_len = np.zeros((self.num_worker_agents,), dtype=np.float32)
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
        # 企业税前利润
        self.firm_pre_tax_profit = np.zeros((self.num_firm_agents,), dtype=np.float32)
        # 各企业所拥有的劳动量
        self.firm_labor = np.zeros((self.num_firm_agents,), dtype=np.float32)
        self.firm_production = np.zeros((self.num_firm_agents,), dtype=np.float32)
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
        - 本期资产
        - 目前工作企业
        - 本期政府税率
        - 上期边际价格
        - 上期各企业工资水平
        - 身份独热编码
        - 上期消费量
        - 上期劳动量
        - 上期报价

        输出维度为(num_worker_agents,worker_obs_dim)
        其中worker_obs_dim=config["size"]["observation"]["worker"]
        """
        worker_obs = np.concatenate([
            self.worker_asset[:, np.newaxis],
            self.worker_in_firm[:, np.newaxis],
            np.full((self.num_worker_agents, 1), self.tax_rate),
            self.firm_wage[np.newaxis, :].repeat(self.num_worker_agents, axis=0),
            self.worker_one_hot,
            self.worker_consumption[:, np.newaxis],
            self.worker_labor[:, np.newaxis],
            self.worker_quote[:, np.newaxis],
        ], axis=-1).astype(np.float32)

        # normalization
        self.rms_worker.update(worker_obs)
        worker_obs = ((worker_obs-self.rms_worker.mean)/np.sqrt(self.rms_worker.var+1e-5)).astype(np.float32)
        return worker_obs

    def construct_firm_obs(self) -> np.ndarray:
        """构造企业的部分观测
        - 上期销量
        - 目前资产
        - 上期税前收入
        - 本期政府税率
        - 上期各企业工资水平
        - 上期各企业报价
        """
        firm_obs = np.concatenate([
            self.firm_asset[:, np.newaxis],
            self.firm_pre_tax_profit[:, np.newaxis],
            self.scalar_repeat(self.tax_rate, self.num_firm_agents)[:, np.newaxis],
            # (num_firm)->(1,num_firm)->(num_firm,num_firm)
            self.firm_wage[np.newaxis, :].repeat(self.num_firm_agents, axis=0),
            self.firm_quote[np.newaxis, :].repeat(self.num_firm_agents, axis=0),
        ], axis=-1).astype(np.float32)

        # normalization
        self.rms_firm.update(firm_obs)
        firm_obs = ((firm_obs-self.rms_firm.mean)/np.sqrt(self.rms_firm.var+1e-5)).astype(np.float32)
        return firm_obs

    def construct_government_obs(self) -> np.ndarray:
        """构造政府观测
        - 各劳动者当前资产
        - 各企业当前资产
        - 各企业当前工资水平
        - 上一期总转移支付
        - 上一期税率
        由于只有一个政府，输出一般是(1,government_obs_dim)
        """
        government_obs = np.concatenate([
            self.worker_asset,
            self.firm_asset,
            self.firm_wage,
            np.array([self.total_transfer]),
            np.array([self.tax_rate])
        ])[np.newaxis, :].astype(np.float32)

        # normalization
        self.rms_government.update(government_obs)
        government_obs = ((government_obs-self.rms_government.mean) /
                          np.sqrt(self.rms_government.var+1e-5)).astype(np.float32)
        return government_obs

    def worker_settlement(self, worker_action: np.ndarray) -> None:
        """劳动者执行后结算：
        - 各个公司的劳动总量
        - 存储劳动者的报价报量，等待市场结算
        - 劳动者跳槽指示更新
          其中action的顺序为：消费，劳动，报价，工作企业


        """
        self.worker_consumption = self.consumption_range[worker_action[:, 0]]
        self.worker_labor = self.labor_range[worker_action[:, 1]]
        self.worker_quote = self.quote_range[worker_action[:, 2]]
        # 跳槽指示
        next_worker_in_firm = worker_action[:, 3]
        self.worker_switch_firm = self.judge_switch_firm(next_worker_in_firm=next_worker_in_firm)
        # 更新劳动者所属企业
        self.worker_in_firm = next_worker_in_firm
        # 各个公司的劳动总量
        self.firm_labor = self.compute_firm_labor()

    def firm_settlement(self, firm_action: np.ndarray) -> None:
        """企业执行动作后结算
        - 计算企业生产量（根据该企业的劳动量，资本量，资本弹性）
        - 市场清算：根据劳动者的报价报量和企业报价和生产量进行匹配：
            - 计算劳动者消费额p_{i,t}*C_{i,t}
            - 计算企业销售额p_{j,t}*C_{j,t}，销量可能来自多个劳动者
        - 工资结算：企业根据劳动量和智能体水平等，发放工资（具体是一个向量和劳动者维度一致）
        """
        self.firm_quote = self.quote_range[firm_action[:, 0]]
        self.firm_wage = self.wage_range[firm_action[:, 1]]
        # 计算企业生产量，根据生产函数

        self.firm_production = np.floor((self.firm_capital**self.firm_capital_elasticity) *
                                        (self.firm_labor**(1-self.firm_capital_elasticity)))

        # 市场清算
        self.market_clearing()
        # 工资结算，对应每个劳动者获得的工资
        self.pre_tax_wages = self.worker_labor*self.worker_levels*self.firm_wage[self.worker_in_firm]
        # 计算企业支付出的工资 TODO：这里其实和compute labor实现一样的
        self.firm_wage_cost = np.bincount(
            self.worker_in_firm, weights=self.pre_tax_wages, minlength=self.num_firm_agents)

    def government_settlement(self, government_action: np.ndarray) -> None:
        """政府执行动作后结算
        - 计算企业效用，即奖励（就是税前利润）
        - 更新企业资产（计算税前利润）
        - 企业资本投入和折旧
        - 企业和劳动者征税
        - 计算转移支付（与资产成反比）
        - 更新劳动者资产（上一期资产+转移支付+税前工资-税收-消费额）*（1+利率）
        - 计算劳动者效用，即奖励（根据消费、劳动、跳槽，资产变化）
        - 计算社会效率
        - 计算社会公平性（根据基尼系数）
        - 计算政府奖励
        """
        self.tax_rate = self.tax_rate_range[government_action[:, 0]].item()
        # 企业税前利润，也是效用和奖励
        self.firm_pre_tax_profit = self.firm_sales-self.firm_wage_cost
        capital_investment = self.firm_asset*self.investment_rate
        capital_investment = np.clip(capital_investment, 0.0, np.inf)
        self.firm_asset = self.firm_asset-capital_investment+(1-self.tax_rate)*self.firm_pre_tax_profit
        self.firm_capital = self.firm_capital*(1-self.depreciation_rate)+capital_investment
        # 征税
        worker_tax = self.pre_tax_wages*self.tax_rate
        firm_tax = self.firm_pre_tax_profit*self.tax_rate
        self.total_transfer = worker_tax.sum()+firm_tax.sum()
        # 转移支付
        weights = inverse_weight_normalized(self.worker_asset)
        transfer_to_worker = self.total_transfer*weights
        # 更新劳动者资产
        new_asset = self.worker_asset+self.pre_tax_wages + \
            transfer_to_worker-worker_tax-self.worker_cost
        # 不允许负债 TODO:有点粗糙，正常来说要避免负债的，后面优化的时候再改
        self.worker_asset = np.clip(new_asset, 0.0, np.inf)

        # 计算劳动者效用（相对风险厌恶为0.1固定，1-0.1=0.9），也是奖励
        self.worker_utility = (self.worker_consumption**0.9)/0.9-self.worker_labor_aversion * \
            self.worker_labor-self.switch_job_penalty*self.worker_switch_firm*self.worker_firm_len

        self.worker_firm_len = (self.worker_firm_len+self.worker_labor)*(1-self.worker_switch_firm)
        self.social_efficiency = np.clip(self.worker_utility.sum(), a_min=0.01, a_max=np.inf)
        self.equality = 1-(self.num_worker_agents)/(self.num_worker_agents-1)*gini(self.pre_tax_wages)
        self.government_reward = ((self.equality)**self.swf_eq_param)*(self.social_efficiency**(1-self.swf_eq_param))
        self.firm_reward = self.firm_pre_tax_profit.copy()
        # 奖励标准化
        self.rms_worker_reward.update(self.worker_utility)
        self.rms_firm_reward.update(self.firm_reward)
        self.rms_government_reward.update(self.government_reward)

        self.worker_utility = ((self.worker_utility-self.rms_worker_reward.mean) /
                               np.sqrt(self.rms_worker_reward.var+1e-5)).astype(np.float32)
        self.firm_reward = ((self.firm_reward-self.rms_firm_reward.mean) /
                            np.sqrt(self.rms_firm_reward.var+1e-5)).astype(np.float32)
        self.government_reward = ((self.government_reward-self.rms_government_reward.mean) /
                                  np.sqrt(self.rms_government_reward.var+1e-5)).astype(np.float32)

    def scalar_repeat(self, scalar: float, n: int) -> np.ndarray:
        "将标量扩展为形状为 (n, ) 的向量。"
        return np.full((n, ), scalar)

    def market_clearing(self):
        """市场清算
        - 所有劳动者申报自己的报价和消费量
        - 所有企业申报自己的报价和生产量
        从最高买家开始匹配最低卖家，为了避免智能体总是出高价保证买到，这里有一个撮合价格
        即成交价实买家卖家出价平均。
        产生一个博弈过程，虽然出高价能让自己优先购买，但也会导致耗费更高。
        卖家可以出低价保证自己先被高价的买，但撮合后也会导致平均后价格低了。

        输入：
        - 劳动者的消费量：self.worker_consumption 维度：(num_worker, )
        - 劳动者的报价：self.worker_quote 维度：(num_worker, )
        - 企业的生产量:self.firm_production 维度：(num_firm, )
        - 企业的报价：self.firm_quote 维度：(num_firm, )

        计算：
        - 真实成交的劳动者消费量，用于后面计算效用
        - 企业的销售额,元素是p_{j,t}*C_{j,t}，用于计算效用和更新资产
        - 劳动者的消费额，用于更新资产

        订单类似：
        {
            0: {'index': 0, 'quantity': 20.0, 'quote': 100.0},
            1: {'index': 3, 'quantity': 10.0, 'quote': 100.0},
            2: {'index': 1, 'quantity': 10.0, 'quote': 90.0},
            3: {'index': 2, 'quantity': 5.0, 'quote': 80.0},
        }
        """
        worker_orders = {
            i: {
                "index": i,
                "quantity": self.worker_consumption[i].item(),
                "quote": self.worker_quote[i].item()
            }
            for i in range(len(self.worker_consumption))
        }
        firm_orders = {
            i: {
                "index": i,
                "quantity": self.firm_production[i].item(),
                "quote": self.firm_quote[i].item()
            }
            for i in range(len(self.firm_production))
        }
        worker_orders = {i: v for i, (_, v) in enumerate(
            sorted(worker_orders.items(), key=lambda item: item[1]['quote'], reverse=True))}
        firm_orders = {i: v for i, (_, v) in enumerate(
            sorted(firm_orders.items(), key=lambda item: item[1]['quote']))}

        worker_index = 0
        firm_index = 0
        worker_consumption = [0]*len(self.worker_consumption)
        firm_sales = [0]*len(self.firm_production)
        worker_cost = [0]*len(self.worker_consumption)
        while worker_index < len(worker_orders) and firm_index < len(firm_orders):
            worker = worker_orders[worker_index]
            firm = firm_orders[firm_index]
            if worker['quote'] >= firm['quote']:
                transaction_price = (firm['quote']+worker['quote'])/2
                transaction_quantity = min(worker['quantity'], firm['quantity'])
                worker_consumption[worker['index']] += transaction_quantity
                cost = transaction_price*transaction_quantity
                firm_sales[firm['index']] += cost
                worker_cost[worker['index']] += cost

                worker_orders[worker_index]['quantity'] -= transaction_quantity
                firm_orders[firm_index]["quantity"] -= transaction_quantity
                if worker_orders[worker_index]['quantity'] < 0.5:
                    worker_index += 1
                if firm_orders[firm_index]["quantity"] < 0.5:
                    firm_index += 1
            else:  # 已经没有可以成交的订单了
                break
        # 计算企业销售额
        self.firm_sales = np.array(firm_sales)
        # 劳动者消费量
        self.worker_consumption = np.array(worker_consumption)
        # 劳动者总开销
        self.worker_cost = np.array(worker_cost)
