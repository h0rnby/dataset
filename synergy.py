import json


class Synergy:
    """
    Class to represent a synergy relationship among a group of projects
    """

    def __init__(self, project_ids, value):
        self.project_ids = project_ids
        self.value = value

    def to_json(self, json_indent=2):
        """
        Convert the project instance to JSON.

        :return: A JSON string
        """
        return json.dumps(self.to_json_dict(), indent=json_indent)

    def to_json_dict(self):
        """
        Convert the Project object to a dictionary that is JSON serializable.

        :return: A dictionary suitable for converting to JSON.
        """

        output = dict()

        output['value'] = self.value
        output['project_ids'] = self.project_ids.tolist()

        return output
