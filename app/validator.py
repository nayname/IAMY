import json
import sys

from config.config import Config
from lib.item import Item
from lib.parse_code import compile_request

class Validator():
    def __init__(self, data, config):
        self.config = Config(...)
        self.data = compile_request(json.loads(data), "local", self.config)
        self.item = Item(self.data, self.config, json.loads(data))

    def validate_data(self):
        self.item.parse_input()

        if self.item.standart_response.is_graded():
            type = self.config.getType()
            ...
        else:
            self.item.response['to_send'] = 0

    def parse_data(self):
        self.item.parse_input()
        return self.item

    def get_response(self):
        self.validate_data()

        self.item.standart_response.set_params()
        ...

        return {...}


if __name__ == "__main__":
    val = Validator(sys.argv[1], 'config/config.json')
    val.get_response()


