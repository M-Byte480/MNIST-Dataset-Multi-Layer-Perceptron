
class House:
    def __init__(self, area, rooms, price):
        self.area = area
        self.rooms = rooms
        self.price = price

    def get_data(self):
        return self.area, self.rooms, self.price