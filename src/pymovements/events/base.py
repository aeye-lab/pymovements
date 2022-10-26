class Event:
    def __init__(self, name: str, onset: int, offset: int):
        self.name = name
        self.onset = onset
        self.offset = offset

    @property
    def duration(self):
        return self.offset - self.onset


class Fixation(Event):
    _name = 'fixation'

    def __init__(self, onset: int, offset: int):
        super().__init__(name=self._name, onset=onset, offset=offset)


class Saccade(Event):
    _name = 'saccade'

    def __init__(self, onset: int, offset: int):
        super().__init__(name=self._name, onset=onset, offset=offset)
