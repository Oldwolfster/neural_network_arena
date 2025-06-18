class TrackInstances:
    instances = []

    def __init__(self):
        self.__class__.instances.append(self)

    def __del__(self):
        try:
            self.__class__.instances.remove(self)
        except ValueError:
            pass  # Object already removed or never properly registered

    @classmethod
    def reset_instances(cls):
        """
        Clears the list of tracked instances.
        Useful when starting a fresh Arena/match.
        """
        cls.instances.clear()

    @classmethod
    def get_all_instances(cls):
        return cls.instances

    @classmethod
    def query_instances(cls, condition):
        """
        Returns a list of instances that match the given condition.

        Args:
            condition (callable): A function that takes an instance and returns True/False.

        Returns:
            List of matching instances.
        """
        return [instance for instance in cls.instances if condition(instance)]
