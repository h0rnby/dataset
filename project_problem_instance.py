import json

import numpy as np

from project_based.instance_parameters import InstanceParameters
from project_based.project import create_random_projects_from_param


class ProjectProblemInstance:
    """
    An object that represents an instance of the project-based PPSSP model.
    """

    def __init__(self, projects, budget, initiation_budget, ongoing_budget, synergies,
                 planning_window, discount_rate, parameters, identifier="Instance"):
        self.projects = projects
        self.budget = budget
        self.initiation_budget = initiation_budget
        self.ongoing_budget = ongoing_budget

        self.synergies = synergies

        self.planning_window = planning_window
        self.identifier = str(identifier)
        self.num_projects = projects.shape[0]
        self.budget_window = budget.shape[0]
        self.discount_rate = discount_rate
        self.parameters = parameters

    def to_json(self, json_indent=2):
        """
        Convert the project instance to JSON.

        :return: A JSON string
        """
        return json.dumps(self.to_json_dict(), indent=json_indent)

    def to_json_dict(self):
        """
        Convert the PortfolioSelectionInstance object to a dictionary that is JSON serializable.

        :return: A dictionary suitable for converting to JSON.
        """
        output = dict()
        output['problem_name'] = self.identifier
        output['periods'] = self.planning_window
        output['budget'] = self.budget.tolist()
        output['initiation_budget'] = self.initiation_budget.tolist()
        output['maintenance_budget'] = self.ongoing_budget.tolist()
        output['synergies'] = [s.to_json_dict() for s in self.synergies]
        output['projects'] = [p.to_json_dict() for p in self.projects]

        return output


def generate_instance(param: InstanceParameters, random_seed, identifier="Instance 1"):
    """
    Generate a random problem instances using the given parameter object and random seed.
    :param param: An InstanceParameters object specifying the problem configuration.
    :param random_seed: The seed used to generate the random list of projects.
    :param identifier: A string that is used as the problem instance name.
    :return:
    """

    # generate a list of random projects and synergy groups
    random_project_list, synergies = create_random_projects_from_param(param, random_seed)

    # find the maximum project length that was generated
    max_length = 0
    for p in random_project_list:
        if p.duration > max_length:
            max_length = p.duration

    param.max_proj_length = max_length

    # calculate the budget period as the planning window plus the maximum project length
    # Note: this is to ensure that a budget is defined if the longest project is selected in the last period
    budget_period = param.planning_window + param.max_proj_length
    budget = np.zeros(budget_period)
    # the budget is increased linearly each year
    for y in range(budget_period):
        budget[y] = param.base_budget + (y * param.yearly_budget_increase)

    # define appropriate proportions for initiation and ongoing budgets
    initiation_budget = budget * param.initiation_max_proportion
    ongoing_budget = budget * param.ongoing_max_proportion

    return ProjectProblemInstance(random_project_list, budget, initiation_budget,
                                  ongoing_budget, synergies, param.planning_window, param.discount_rate, param,
                                  identifier)
