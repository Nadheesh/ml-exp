"""
Copyright 2018 Nadheesh Jihan

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import pandas as pd


def write_to_excel(df: pd.DataFrame, file_name, sheet_name):
    """
    write pandas dataframe to a excel file, creating a new sheet
    :param df:
    :param file_name:
    :param sheet_name:
    :return:
    """
    file = pd.ExcelWriter('%s.xlsx' % file_name)
    df.append(df.to_excel(file, sheet_name=sheet_name))
    file.close()
