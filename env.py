from utils import load_config, generate_levels, distribute_evenly, worker_one_hot
import torch


class EconomicEnv:
    def __init__(self):
        self.config = load_config()
        self.num_worker_agents = self.config['num_worker_agents']
        self.num_firm_agents = self.config['num_firm_agents']
        self.interest_rate = self.config['constants']['interest_rate']
        self.device = self.config['device']
        # 所有可能的报价
        self.quote = self.config['constants']['quote']
        self.labor_range = self.config['constants']['labor_range']
        self.consumption_range = self.config['constants']['consumption_range']

    def reset(self):
        # ========================== 劳动者相关 ==========================
        # 劳动者资产
        self.worker_asset = torch.full((self.num_worker_agents,),
                                       self.config['initialize']['worker_asset']).to(self.device)
        # 劳动者技能禀赋
        self.worker_levels = generate_levels(self.num_worker_agents)
        # 劳动者报价
        self.worker_quote = torch.full((self.num_worker_agents,),
                                       self.config['initialize']['quote']).to(self.device)
        # 初始化每个劳动者所属的企业
        self.worker_in_firm = torch.tensor(distribute_evenly(
            self.num_worker_agents, self.num_firm_agents), dtype=torch.long).to(self.device)
        # 劳动者独热编码
        self.worker_one_hot = worker_one_hot(self.num_worker_agents).to(self.device)
        # 劳动者劳动量
        self.worker_labor = torch.zeros((self.num_worker_agents,)).to(self.device)
        # 劳动者消费量
        self.worker_consumption = torch.zeros((self.num_worker_agents,)).to(self.device)
        # ========================== 企业相关 ==========================
        # 企业资产
        self.firm_asset = torch.full((self.num_firm_agents,),
                                     self.config['initialize']['firm_asset']).to(self.device)
        # 企业资本
        self.firm_capital = torch.full((self.num_firm_agents,),
                                       self.config['initialize']['firm_capital']).to(self.device)
        # 各企业报价
        self.firm_quote = torch.full((self.num_firm_agents,),
                                     self.config['initialize']['quote']).to(self.device)
        # 各企业工资水平
        self.firm_wage = torch.full((self.num_firm_agents,),
                                    self.config['initialize']['wage']).to(self.device)
        self.firm_sell = torch.zeros((self.num_firm_agents,)).to(self.device)
        # 企业税前利润
        self.firm_pre_tax_profit = torch.zeros((self.num_firm_agents,)).to(self.device)
        # ========================== 政府相关 ==========================
        # 政府税率
        self.tax_rate = self.config['initialize']['tax_rate']
        # 上一期总转移支付（税收）
        self.total_transfer = 0

        # ========================== 其他参数 ==========================
        # 边际价格
        self.marginal_price = self.config['initialize']['marginal_price']

    def compute_firm_labor(self):
        """根据劳动者的劳动量，计算每个企业的总劳动量
        例如劳动量向量是[100,200,300,400,100]，
        对应工作企业序号向量是[0,1,1,2,0],
        输出为和企业数量维度一致的向量，代表在每家企业的总劳动量
        [200,500,400]
        """
        return torch.bincount(self.worker_in_firm, weights=self.worker_labor, minlength=self.num_firm_agents)

    def judge_switch_firm(self, next_worker_in_firm):
        """判断劳动者是否跳槽，若跳槽则对应向量位置为1"""
        return torch.where(self.worker_in_firm != next_worker_in_firm,
                           torch.ones(self.num_worker_agents, dtype=torch.long),
                           torch.zeros(self.num_worker_agents, dtype=torch.long)).to(self.device)
