class InstanceParameters:
    """
    A wrapper object used to specify control parameters for a problem instance.
    """

    def __init__(self,
                 num_projects,
                 planning_window,
                 base_budget,
                 yearly_budget_increase,
                 initiation_max_proportion=0.25,
                 ongoing_max_proportion=0.75,
                 prerequisite_tuples=None,
                 exclusion_tuples=None,
                 synergy_tuples=None,
                 discount_rate=0.0,
                 **kwargs):
        self.num_projects = num_projects
        self.planning_window = planning_window
        self.base_budget = base_budget
        self.yearly_budget_increase = yearly_budget_increase
        self.initiation_max_proportion = initiation_max_proportion
        self.ongoing_max_proportion = ongoing_max_proportion
        self.prerequisite_tuples = prerequisite_tuples
        self.exclusion_tuples = exclusion_tuples
        self.synergy_tuples = synergy_tuples
        self.discount_rate = discount_rate
        self.kwargs = kwargs
