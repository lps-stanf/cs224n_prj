from configparser import ConfigParser
from pydoc import locate
import argparse


class SettingsKeeper(object):
    def __init__(self):
        self._settings_dict = {}

    def __getattr__(self, name):
        if name not in self._settings_dict:
            raise AttributeError('No such setting: {0}'.format(name))
        return self._settings_dict.get(name)

    def add_dictionary(self, dict):
        for k, v in dict.items():
            self._settings_dict[k] = v

    def _add_key_value(self, key, val, type=None):
        if type is not None:
            val = type(val)
        self._settings_dict[key] = val

    def add_key_from_dict(self, dict, key, type=None, default=None):
        val = dict.get(key, default)
        self._add_key_value(key, val, type)

    def add_ini_file(self, ini_file_path):
        config = ConfigParser(allow_no_value=True)
        with open(ini_file_path, 'r') as f:
            config.read_file(f)

        for section in config.sections():
            for option_key in config.options(section):
                option_value = config.get(section, option_key)
                option_key_list = option_key.split()
                if len(option_key_list) > 2:
                    raise ValueError('Error in config, key is too long "{}"'.format(option_key))
                type = None
                if len(option_key_list) == 2:
                    type = locate(option_key_list[0])
                    option_key_list.pop(0)
                self._add_key_value(option_key_list[0], option_value, type)

    def add_parsed_arguments(self, args: argparse.Namespace):
        for prop, val in vars(args).items():
            self._add_key_value(prop, val)