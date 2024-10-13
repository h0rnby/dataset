import json

import numpy as np
import scipy.stats as dists

from project_based.datagen import mvlnorm_generate_costdur, fuzzy_weibull_cost_distribution
from project_based.synergy import Synergy


class Project:
    """
    A project in the project-based PPSSP model.
    """

    def __init__(self, project_id, project_name, cost, value, duration, total_cost,
                 prerequisite_list, successor_list, exclusion_list):
        self.id = project_id
        self.project_name = project_name
        self.cost = cost
        self.value = value
        self.duration = duration
        self.prerequisite_list = prerequisite_list
        self.successor_list = successor_list
        self.exclusion_list = exclusion_list
        self.total_cost = total_cost

    def __str__(self):
        return f"\tID: {self.id}" \
               f"\tName: {self.project_name}" \
               f"\tCost:{self.nparray_tostring_helper(self.cost)}" \
               f"\tValue:{self.nparray_tostring_helper(self.value)}" \
               f"\tDur:{self.duration}" \
               f"\tPred.:{self.nparray_tostring_helper(self.prerequisite_list)}" \
               f"\tSucc.:{self.nparray_tostring_helper(self.successor_list)}" \
               f"\tExcl.:{self.nparray_tostring_helper(self.exclusion_list)}"

    def __repr__(self):
        return self.__str__()

    def nparray_tostring_helper(self, array):
        """
        Helper method to print a numpy array on one line
        :param array: A numpy array
        :return: String representation of the numpy array.
        """
        return np.array2string(array).replace('\n', '')

    def to_json(self, json_indent=2):
        """
        Convert the project to JSON.

        :return: A JSON string
        """
        return json.dumps(self.to_json_dict(), indent=json_indent)

    def to_json_dict(self):
        """
        Convert the Project object to a dictionary that is JSON serializable.

        :return: A dictionary suitable for converting to JSON.
        """
        output = dict()
        output['id'] = self.id
        output['name'] = self.project_name
        output['duration'] = int(self.duration)
        output['prerequisites'] = self.prerequisite_list.tolist()
        output['successors'] = self.successor_list.tolist()
        output['mutual_exclusions'] = self.exclusion_list.tolist()
        output['cost'] = self.cost.tolist()
        output['value'] = self.value.tolist()

        return output


def create_random_projects_from_param(params, seed=1):
    """
    Create a list of random projects using a parameter object to specify the configuration.

    :param params: An InstanceParameters object specifying the project generation configuration.
    :param seed: The random seed

    :return: A numpy array of Project objects
    """
    return create_random_projects(params.num_projects,
                                  params.prerequisite_tuples,
                                  params.exclusion_tuples,
                                  params.synergy_tuples,
                                  seed,
                                  **params.kwargs)


def create_random_projects(num_projects,
                           prerequisite_tuples=None,
                           exclusion_tuples=None,
                           synergy_tuples=None,
                           seed=1,
                           **kwargs):
    """
    Create a list of random projects.

    :param num_projects: Number of projects
    :param prerequisite_tuples: Tuples representing the number and chance of a project having prerequisite(s).
    The tuple (g, p) denotes that g prerequisites will be generated for proportion p of projects.
    :param exclusion_tuples: Tuples representing the number and chance of a project having mutual exclusion(s).
    The tuple (g, p) denotes that a mutual exclusion group of size g will be generated for proportion p of projects.
    :param synergy_tuples: Tuples representing the number and chance of a project forming a synergy group.
    The tuple (g, p) denotes that a proportion p of the projects will exist in a synergy group of size g.
    :param seed: The random seed
    :param kwargs: Other keyword arguments

    :return: A tuple containing a numpy array of Project objects and a list of Synergy objects.
    """
    np.random.seed(seed)  # scipy uses numpy to generate random
    projects = np.empty(num_projects, dtype=object)

    # get the (total) cost and duration form the multi-variate normal distribution
    cost_dur = mvlnorm_generate_costdur(num_projects)

    for i in range(num_projects):
        duration = cost_dur[i, 0]
        total_cost = cost_dur[i, 1]
        # distribute the cost over each year
        cost = fuzzy_weibull_cost_distribution(total_cost, duration)

        # generate the total value and distribute it over each year
        total_value = random_cost_dur_value(total_cost, duration, **kwargs)
        value = fuzzy_weibull_cost_distribution(total_value, duration)
        value = value.round()

        # define empty lists for mutual exclusions, prerequisites, and successors. These are filled after all projects
        # have been generated
        exclusions = np.empty(0, dtype=int)
        prerequisites = np.empty(0, dtype=int)
        successors = np.empty(0, dtype=int)

        projects[i] = Project(i + 1, f"Project {i + 1}", cost, value, duration, total_cost,
                              prerequisites, successors, exclusions)

    # generate the mutual exclusion constraints
    if exclusion_tuples is not None:
        _generate_exclusions_post(projects, exclusion_tuples)

    # generate the prerequisite relationships
    if prerequisite_tuples is not None:
        _generate_prerequisites_post(projects, prerequisite_tuples)

    # generate the synergy groups
    synergy_groups = []
    if synergy_tuples is not None:
        synergy_groups = _generate_synergies_post(projects, synergy_tuples)

    return projects, synergy_groups


def random_cost_dur_value(total_cost, duration, **kwargs):
    """
    Generate the value for a project based on its total cost and duration.

    :param total_cost: The total cost associated with a project.
    :param duration: The duration of the project.
    :param kwargs: Any other keyword arguments, which can be used to modify the value distribution and factor.

    :return: The randomly-generated value for a project with the given cost and duration.
    """
    value_dist = kwargs["value_dist"] if "value_dist" in kwargs else dists.randint(1, 5)
    factor = kwargs["factor"] if "factor" in kwargs else 2

    return np.random.random() * total_cost * factor + sum(value_dist.rvs(duration - 1))


def _generate_prerequisites_post(projects, group_sizes):
    """
    Generate the list of prerequisite projects after the entire set of projects has been generated. This DOES NOT
    into account any mutually exclusive projects, thus impossible constraints may be generated.

    :param projects: The list of projects
    :param group_sizes: A list of tuples [(g1, p1), ...., (gn, pn)], where gn represents the group
    size and pn represents the probability for that group size. For example [(2, 0.05)] indicates that a precendence
    constraint of size 2 (i.e., one successor for the prerequisite) will occur on approximately 5% of projects,
    while [(3, 0.05)] indicates that a precendence constraint of size 3 (i.e., two successors for the prerequisite)
    will occur on approximately 5% of projects.

    :return: None. Projects are directly modified.

    """
    num_projects = projects.shape[0]
    indices = np.arange(num_projects)

    for size, prop in group_sizes:
        num_groups = int(prop * num_projects / size)

        groups = np.random.choice(indices, size=(num_groups, size), replace=False)

        # remove projects from being selected in future rounds
        indices = np.setdiff1d(indices, groups)

        for group in groups:
            # sort group, assuming that index i < j indicates project[i] is a prerequisite for project[j]
            sorted_group = np.sort(group)
            prereq = sorted_group[0]  # first element is the prerequisite

            for i in range(1, size):  # remaining elements are successors
                successor = sorted_group[i]
                projects[prereq].successor_list = np.append(projects[prereq].successor_list, successor + 1)
                projects[successor].prerequisite_list = np.append(projects[successor].prerequisite_list, prereq + 1)


def _generate_exclusions_post(projects, group_sizes):
    """
    Generate the list of mutually exclusive projects after the entire set of projects has been generated.
    This DOES NOT take into account any prerequsite relationships and thus impossible constraints may be generated.

    :param projects: The list of projects.
    :param group_sizes: A list of tuples [(g1, p1), ...., (gn, pn)], where gn represents the group size and pn represents
    the probability for that group size. For example [(2, 0.05)] indicates that a mutually exclusive group of size 2 (i.e.,
    one prerequisite) will occur on approximately 5% of projects while [(2, 0.05), (3, 0.10)] indicates that a mutually
    exclusive group of size 2 (i.e., one prerequisite) will occur on approximately 5% of projects and a group of size 3
    will occur on approximately 10% of projects.

    :return: None. Projects are directly modified.
    """
    num_projects = projects.shape[0]
    indices = np.arange(num_projects)

    for size, prop in group_sizes:
        num_groups = int(prop * num_projects / size)

        groups = np.random.choice(indices, size=(num_groups, size), replace=False)

        # remove projects from being selected in future rounds
        indices = np.setdiff1d(indices, groups)

        for group in groups:
            for i in range(size):
                ind1 = group[i]
                # add all projects (other than i) to exclusion list
                for j in range(size):
                    if i == j:
                        continue
                    ind2 = group[j]
                    projects[ind1].exclusion_list = np.append(projects[ind1].exclusion_list, ind2 + 1)


def _generate_synergies_post(projects, group_sizes):
    """
    Generate the list of synergies after the entire set of projects has been generated.
    This DOES take into account any mutual exclusion relationships.

    :param projects: The list of projects.
    :param group_sizes: A list of tuples [(g1, p1), ...., (gn, pn)],
    where gn represents the group size and pn represents the probability for that group size. For example [(2,
    0.05)] indicates that a synergy group of size 2  will occur on approximately 5% of projects while [(2, 0.05), (3,
    0.10)] indicates that a synergy group of size 2 (i.e., one prerequisite) will occur on approximately 5% of
    projects and a group of size 3 will occur on approximately 10% of projects.

    :return: A list of synergy objects, which each contain a numpy array of project ids and a corresponding value.
    """
    num_projects = projects.shape[0]
    indices = np.arange(num_projects)

    synergies = []

    for size, prop in group_sizes:
        num_groups = int(prop * num_projects / size)

        for i in range(num_groups):
            # generate a random group and verify that there are no exclusion . If issues, regenerate
            repeat = True
            while repeat:
                repeat = False
                # generate a random group of projects
                group = np.random.choice(indices, size=size, replace=False)
                # for each project, verify its exclusions do not prevent this synergy from being realized
                for index, p_index in enumerate(group):
                    if repeat:  # hack to exit for loop early if repeat is necessary
                        break
                    p_check = projects[p_index]
                    # check only projects that succeed this project in the group as exclusions are mutual
                    for check_index in range(index + 1, size):
                        p2_index = group[check_index]
                        # if p2 is in the exclusion list of p1, this group will be regenerated
                        if p2_index in p_check.exclusion_list:
                            repeat = True
                            break  # exit to outer loop, since we no longer need to verify any further

            # generate a random value for the synergy group as a random value between 1 and the total (non-discounted)
            # value of the projects within the group
            total_group_value = 0
            for p_index in group:
                total_group_value += np.sum(projects[p_index].value)
            synergy_value = np.random.randint(1, total_group_value)

            synergies.append(Synergy(group, synergy_value))

    return synergies


def roulette_wheel_select(candidates: np.array, weights: np.array, alpha=None):
    """
    Implementation of a roulette wheel selection with given weights.
    :param candidates: The candidates to select from
    :param weights: The weights associated with each candidate
    :param alpha: An optional exponent to scale the weight value for each candidate. Default: None
    :return: The candidates selected via roulette wheel selection.
    """
    if alpha is None:
        p = weights / weights.sum()
    else:
        weight_scaled = np.power(weights, alpha)
        p = weight_scaled / weight_scaled.sum()

    return np.random.choice(candidates, p=p)
