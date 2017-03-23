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

    def __setattr__(self, key, value):
        if key == '_settings_dict':
            object.__setattr__(self, key, value)
        else:
            if key not in self._settings_dict:
                raise AttributeError('No such setting: {0}'.format(key))
            self._settings_dict[key] = value

    def add_dictionary(self, dict):
        for k, v in dict.items():
            self._settings_dict[k] = v

    def add_key_value(self, key, val, type=None):
        if type is not None:
            val = type(val)
        self._settings_dict[key] = val

    def add_key_from_dict(self, dict, key, type=None, default=None):
        val = dict.get(key, default)
        self.add_key_value(key, val, type)

    def _add_ini_file_section(self, config_parser, section_name, require_provided_section=True):
        sections_list = config_parser.sections()
        if section_name not in sections_list:
            if not require_provided_section:
                return
            raise RuntimeError('No required section in config file: "{0}"'.format(section_name))

        for option_key in config_parser.options(section_name):
            option_value = config_parser.get(section_name, option_key)
            option_key_list = option_key.split()
            if len(option_key_list) > 2:
                raise ValueError('Error in config, key is too long "{}"'.format(option_key))
            type = None
            if len(option_key_list) == 2:
                type = locate(option_key_list[0])
                option_key_list.pop(0)
            self.add_key_value(option_key_list[0], option_value, type)

    def add_ini_file(self, ini_file_path, sections_list=None, require_provided_sections=True):
        config_parser = ConfigParser(allow_no_value=True)
        with open(ini_file_path, 'r') as f:
            config_parser.read_file(f)

        if sections_list is not None:
            for section in sections_list:
                self._add_ini_file_section(config_parser, section, require_provided_sections)
        else:
            for section in config_parser.sections():
                self._add_ini_file_section(config_parser, section)

    def add_parsed_arguments(self, args: argparse.Namespace):
        for prop, val in vars(args).items():
            self.add_key_value(prop, val)
