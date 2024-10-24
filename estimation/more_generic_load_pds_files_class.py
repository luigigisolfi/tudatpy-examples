#!/usr/bin/env python
# coding: utf-8

# ## Class for Loading PDS files

import sys
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
from urllib.request import urlretrieve
from datetime import datetime, timedelta
import glob
import re

class LoadPDS:

    ########################### SECTION START: COMMON FUNCTIONS FOR ALL TYPES OF FILES ######################################
    
    def __init__(self):
        self.Orientation = self.Orientation(self)  # Instantiate Orientation class
        self.IFMS = self.IFMS(self) 

    def generic_strptime(self, date):

        """ 
        Given a list of supported time formats, try to convert it with strptime.
        This function is used throughout the code. 
        
        For instance: 
          juice and mro orientation files support %y%m%d,
          lro supports %Y%j,
          other mro kernels might support %Y_%j,
          etc...
        """

        self.supported_time_formats = {
            "YYMMDD": (
                r"%y%m%d"
            ),
            "YYYYjjj": (
                r"%Y%j"
            ),   
            "YYYY_jjj": (
                r"%Y_%j"
            ),   
            "YYjjjhhmm": (
                r"%y%j%H%M"
            ),
            "YYYY_JJJ_hhmm":(
            r"%Y_%j_%H%M"
            )
        }

        for key, str_format in self.supported_time_formats.items():
            try: 
                self.date = datetime.strptime(date,str_format )
                if self.date is not None:
                    return (self.date)
            except:
                continue      

    def add_custom_mission_pattern(self, input_mission, custom_pattern):
        
        """
        THE IDEA OF THIS FUNCTION IS TO ALLOW FOR USERS TO DEFINE THEIR OWN CUSTOMISED PATTERN.
        ONCE THE PATTERN IS DEFINED, this can be added to the supported ones.
        """
        custom_dict = {}
        custom_dict[input_mission] = (custom_pattern, input_mission)
        self.Orientation.supported_orientation_formats.update(custom_dict)
        
        return self.Orientation.supported_orientation_formats

    def is_date_in_intervals(self, date, intervals):
        self.in_intervals = False
        for interval in intervals:
            if ( ( date.date() >= interval[0].date() ) and ( date.date() <= interval[1].date() ) ):
                self.in_intervals = True
        return (self.in_intervals)
        

    def dynamic_download_url_files_time_interval(self, input_mission, local_path, filename_format, start_date, end_date, url, time_interval_format): #TAKES INPUT_MISSION (NAME) as INPUT
    
        # Prepare the date range for searching existing files
        all_dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
        existing_files = glob.glob(local_path + filename_format)
        relevant_intervals = []
        self.relevant_files = []    
        
        for full_filename in existing_files:
            filename = full_filename.split("/")[1]
            orientation_dict = self.Orientation.parse_orientation_filename(input_mission, filename)
            filename_to_download = self.Orientation.reconstruct_orientation_filename(input_mission , orientation_dict)  
            start_time = orientation_dict["start_date_utc"]
            end_time = orientation_dict["end_date_utc"]

            is_relevant = False
            date_cnt = 0
            current_interval = (start_time,end_time)
            while not is_relevant and date_cnt < len(all_dates):
                if self.is_date_in_intervals(all_dates[date_cnt], [current_interval]):
                    is_relevant = True
                    relevant_intervals.append(current_interval)
                    self.relevant_files.append(full_filename)
                date_cnt += 1

        dates_without_file = [date for date in all_dates if not self.is_date_in_intervals(date, relevant_intervals)]
    
        if dates_without_file: 
            reqs = requests.get(url)
            files_url_dict = {}
    
            for link in BeautifulSoup(reqs.text, 'html.parser').find_all('a'):
                full_link = link.get('href')
                if isinstance(full_link, str):
                    print('Full link:', full_link)  # Output the full link for inspection
            
                    # Check if the link is a full URL or a relative path
                    if ('/') in full_link:
                        # Full URL
                        filename = os.path.basename(full_link)
                        print("This is a full URL.")
                    else:
                        # Relative path
                        filename = full_link
                        print("This is a relative path.")
                        
                    print(filename)
                    print(filename_format.split('*')[0])
                    print(filename_format.split('*')[1])
                    if filename.startswith(filename_format.split('*')[0]) and filename.endswith(filename_format.split('*')[1]):
                        try:
                            orientation_dict = self.Orientation.parse_orientation_filename(input_mission, filename)
                            print(orientation_dict)
                        except:
                            print(f'Could not download orientation file: {filename}')
                            continue
                    else:
                        continue
                    filename_to_download = self.Orientation.reconstruct_orientation_filename(input_mission,orientation_dict)
                    start_time = orientation_dict["start_date_utc"]
                    end_time = orientation_dict["end_date_utc"]   
                    
                    files_url_dict[(start_time, end_time)] = filename

            for date in dates_without_file:
                if not self.is_date_in_intervals(date, relevant_intervals):
                    for new_interval in files_url_dict:
                        if self.is_date_in_intervals(date, [new_interval]):
                            print('Downloading', files_url_dict[new_interval])
                            urlretrieve(url + files_url_dict[new_interval], local_path + files_url_dict[new_interval])
                            self.relevant_files.append(local_path + files_url_dict[new_interval])
                            relevant_intervals.append(new_interval)
        print('Done.')    
        return self.relevant_files


    def dynamic_download_url_files_single_time(self, input_mission, local_path, filename_format, start_date, end_date, url, time_format):
        
        # Determine the file type
        if ".tab" in filename_format:
            file_type = "IFMS"
            print('file extension is .tab')
        elif ".odf" in filename_format:
            file_type = "ODF"
        else:
            raise ValueError("Invalid Radio Science File Type (must be .ODF or .IFMS)")
    
        # Check date range validity
        if start_date > end_date:
            raise ValueError("start_date must be before end_date.")
    
        # Prepare the date range for searching existing files
        all_dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
        existing_files = glob.glob(os.path.join(local_path, filename_format))
        
        self.relevant_files = []                            
    
        # If there are existing files, process them
        if existing_files:
            for full_filename in existing_files:
                filename = os.path.basename(full_filename)  # Get the filename from the full path
            
                # Generate the expected filename
                if file_type == "IFMS":
                    try:
                        IFMS_dict = self.IFMS.parse_IFMS_filename(input_mission, filename)
                        filename_to_download = self.IFMS.reconstruct_IFMS_filename(input_mission, IFMS_dict)
                    except Exception as e:
                        print(f"Error parsing IFMS filename {filename}: {e}")
                        continue
                elif file_type == "ODF":
                    try:
                        IFMS_dict = self.ODF.parse_ODF_filename(input_mission, filename)
                        filename_to_download = self.ODF.reconstruct_ODF_filename(input_mission, IFMS_dict)
                    except Exception as e:
                        print(f"Error parsing ODF filename {filename}: {e}")
                        continue
            
                print('Expected filename to download:', filename_to_download)
    
                # Check if the file already exists
                if filename in [os.path.basename(x) for x in existing_files]:
                    self.relevant_files.append(full_filename)  # Add the matching file
                else:
                    print('File not found, will download:', filename_to_download)
    
        # If there are no existing files, we need to download all required files
        try:
            reqs = requests.get(url)
            reqs.raise_for_status()  # Raise an error for bad responses
    
            files_url_dict = {}
    
            # Retrieve files from the server
            for link in BeautifulSoup(reqs.text, 'html.parser').find_all('a'):
                full_link = link.get('href')
                if isinstance(full_link, str):
                    print('Full link:', full_link)  # Output the full link for inspection
            
                    # Check if the link is a full URL or a relative path
                    if ('/') in full_link: #on pds, two things happen when you click on a link 1) download of file starts (relative url) or 2) link (full url) opens. 
                        # Full URL
                        filename = os.path.basename(full_link)
                        print("This is a full URL.")
                    else:
                        # Relative path
                        filename = full_link
                        print("This is a relative path.")
                    print(filename)
                    print(filename_format.split('*')[0])
                    print(filename_format.split('*')[1])
                    if filename.startswith(filename_format.split('*')[0]) and filename.endswith(filename_format.split('*')[1]):
                            
                        try:
                            orientation_dict = self.IFMS.parse_IFMS_filename(input_mission, filename)
                            print(orientation_dict)
                            filename_to_download = self.IFMS.reconstruct_IFMS_filename(input_mission, orientation_dict)
                            print(filename_to_download)

                            if input_mission == 'MEX' or input_mission == 'mex':
                                files_url_dict[orientation_dict["date_file"][:5]] = filename_to_download #MEX: YYJJJHHMM, we only care about YYJJJ
                            else:
                                files_url_dict[orientation_dict["date_file"][:8]] = filename_to_download #THESE SHOULD BE FINE
                        except Exception as e:
                            print(f'Could not parse orientation file: {filename} - Error: {e}')
                            continue
    
            # Download the missing files
            for date in all_dates:
                if input_mission == 'MEX' or input_mission == 'mex': #Deals with MEX: YYJJJHHMM format
                    year = date.year
                    day_of_year = date.timetuple().tm_yday
                    # Format components
                    date_string = f"{year % 100:02d}{day_of_year:03d}"
                else:
                    date_string = date.strftime(time_format) #Deals with NON-MEX
                    print(date_string)
                    print(files_url_dict)

                if date_string in files_url_dict:
                    download_file = files_url_dict[date_string]
                    full_download_url = os.path.join(url, download_file)
                    print('Downloading', full_download_url)
    
                    try:
                        urlretrieve(full_download_url, os.path.join(local_path, download_file))
                        self.relevant_files.append(os.path.join(local_path, download_file))
                    except Exception as e:
                        print(f"Failed to download {full_download_url}: {e}")
    
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {url}: {e}")

        print('Done.')
        return self.relevant_files

    
    ########################### SECTION END: COMMON FUNCTIONS FOR ALL TYPES OF FILES ######################################


    ########################### SECTION START: IFMS CLASS ######################################

    class IFMS:
        def __init__(self, LoadPDS):
            self.supported_IFMS_formats = {       
                "mex":
                    r'^(?P<mission>[a-zA-Z0-9]+)_(?P<band>[a-zA-Z0-9]+)_(?P<date_file>[0-9]{9})_(?P<version>[0-9]{2})(?P<extension>\.tab$)',
                "mro":
                    r'^(?P<mission>mro)(?P<dataset>magr)(?P<date_file>\d{4}_\d{3}_\d{4})(?P<uplink>x|n)(?P<station>nn|mm)(?P<downlink>[123m])(?P<version>v\d+)(?P<extension>\.tnf|.odf)$'
            }
        def parse_IFMS_filename(self, input_mission, filename): #TAKES INPUT_MISSION (NAME) as INPUT

            """ 
            basically all datasets have: mission name, instrument, purpose (optional), 
            start date, end date, leapsecond file (optional), version (optional), extension (.bc for orientation). 
            idea: parse each pattern for each mission (so we need to know exactly how the pattern is, 
            for instance for juice and mro separately) and create a dictionary with both optional and non optional keys. 
            This is of course only done for supported missions. 
            """
            
            IFMS_dict = None
            list_juice_mro_lro = ['JUICE', 'juice', 'mro', 'MRO', 'LRO', 'lro']
            
            # Attempt to retrieve the pattern for the specified mission directly
            if input_mission in self.supported_IFMS_formats:
                pattern = self.supported_IFMS_formats[input_mission]
                match = re.match(pattern, filename)
            else:
                raise ValueError('Selected Mission Not Supported (yet!) Aborting ...') 

            if match:
                IFMS_dict = match.groupdict()
                print(IFMS_dict)
                
                self.date_utc = LoadPDS.generic_strptime(LoadPDS, IFMS_dict["date_file"])          
                IFMS_dict["date_utc"] = self.date_utc
                return (IFMS_dict)
                              
            if IFMS_dict == None:
                raise ValueError("Could not create IFMS_dict")
                
        def get_mission_IFMS_expected_keys(self, input_mission):
            """
            Returns the expected keys for the given mission.
            
            Parameters:
            - input_mission (str): The name of the mission.
            
            Returns:
            - list: The list of expected keys for the mission.
            """
            if input_mission == "mex":
                return ['mission', 'band', 'station', 'data_souce_id', 'data_archiving_level', 'date_file', 'version', 'extension']
            elif input_mission == "mro":
                return ['mission', 'dataset', 'date_file', 'uplink', 'station', 'downlink', 'version', 'extension']
                
            # Add more missions and their expected keys as needed
            else:
                return []


        def join_IFMS_keys(self, input_mission, dictionary):
            
            """
            Formats the expected keys based on the mission's rules.
        
            Parameters:
            - input_mission (str): The name of the mission.
            - expected_keys (list): The list of keys to format.
        
            Returns:
            - str: The formatted string based on the mission's requirements.
            """
            values = list(dictionary.values())
                    
            if input_mission == "mex":
                mission, band, date_file, version, extension, date_utc = values
                
                filename = f'{mission}_{band}_{date_file}_{version}{extension}'
                return filename  # Using underscores
            elif input_mission == "mro":
                mission, dataset, date_file, uplink, station, downlink, version, extension, date_utc = values
                filename = f'{mission}{dataset}{date_file}{uplink}{station}{downlink}{version}{extension}'
                return filename
            # Add more missions with specific formatting rules as needed
            else:
                return "_".join(expected_keys)  # Default behavior

        def reconstruct_IFMS_filename(self, input_mission, IFMS_dict): 
            """
            Reconstructs the original filename from the IFMS dictionary.
            
            Parameters:
            - input_mission (str): The name of the mission.
            - IFMS_dict (dict): The dictionary containing the file information.
            
            Returns:
            - str: The reconstructed filename.
            """
            
            # Join the parts with underscores
            self.IFMS_filename = self.join_IFMS_keys(input_mission, IFMS_dict)
        
            return self.IFMS_filename
            
    ########################### SECTION END: IFMS CLASS ######################################
                
    ########################### SECTION START: ORIENTATION CLASS ######################################                
    class Orientation:
    
        def __init__(self, LoadPDS):
            self.supported_orientation_formats = {
                "juice":
                    "^(?:([a-z]+)_)?(?:([a-z]+)_)?(?:([a-z]+)_)?(\\d{6})_(\\d{6})(?:_s(\\d{6}))?(?:_(v\\d+))?(\.bc)$",
                "mex":
                    "^(?P<data>ATNM)_(?P<purpose>MEASURED)_(?P<start_date_file>\d{6})_(?P<end_date_file>\d{6})_(?P<version>(V\d+))?(?P<extension>\.BC)$",            
                
                "mro":
                     r"^(?P<mission>mro)_(?P<instrument>[a-z]+)(?:_(?P<purpose>[a-z]+))?_(?P<start_date_file>\d{6})_(?P<end_date_file>\d{6})(?P<extension>\.bc)$",   
                "lro":
                    r"^(lro)([a-z]+)?(?:_([a-z]+))?_(\d{6}|\d{7})_(\d{6}|\d{7})(?:_(s\d{6}))?(?:_(v\d+))?(\.bc)$",
                "cassini":
                    r"^(?:([a-zA-Z0-9]+)_)?(\d{5})_(\d{5})([a-z]{0,2})?(\.bc)$"   
            }
        
        def parse_orientation_filename(self, input_mission, filename): #TAKES INPUT_MISSION (NAME) as INPUT
    
            """ 
            basically all datasets have: mission name, instrument, purpose (optional), 
            start date, end date, leapsecond file (optional), version (optional), extension (.bc for orientation). 
            idea: parse each pattern for each mission (so we need to know exactly how the pattern is, 
            for instance for juice and mro separately) and create a dictionary with both optional and non optional keys. 
            This is of course only done for supported missions. 
            """
        
            # Generic pattern for both JUICE and MRO orientation formats
            orientation_dict = None
            #print(self.supported_orientation_formats.items())
            list_juice_mro_lro = ['JUICE', 'juice', 'mro', 'MRO', 'LRO', 'lro']

            # Attempt to retrieve the pattern for the specified mission directly
            if input_mission in self.supported_orientation_formats:
                pattern = self.supported_orientation_formats[input_mission]
                match = re.match(pattern, filename)
            else:
                raise ValueError('Selected Mission Not Supported (yet!) Aborting ...') 

            if match:
                orientation_dict = match.groupdict()
                print(orientation_dict)
                
                self.start_date_utc = LoadPDS.generic_strptime(LoadPDS, orientation_dict["start_date_file"]) 
                self.end_date_utc = LoadPDS.generic_strptime(LoadPDS, orientation_dict["end_date_file"])  
                orientation_dict["start_date_utc"] = self.start_date_utc
                orientation_dict["end_date_utc"] = self.end_date_utc
                return (orientation_dict)
                              
            if orientation_dict == None:
                raise ValueError("Could not create orientation_dict")
                    
        def reconstruct_orientation_filename(self, input_mission, orientation_dict): 
            """
            Reconstructs the original filename from the orientation dictionary.
            
            Parameters:
            - input_mission (str): The name of the mission.
            - orientation_dict (dict): The dictionary containing the file information.
            
            Returns:
            - str: The reconstructed filename.
            """
            
            # Join the parts with underscores
            self.orientation_filename = self.join_orientation_keys(input_mission, orientation_dict)
        
            return self.orientation_filename
            
        def join_orientation_keys(self, input_mission, dictionary):
            
            """
            Formats the expected keys based on the mission's rules.
        
            Parameters:
            - input_mission (str): The name of the mission.
            - expected_keys (list): The list of keys to format.
        
            Returns:
            - str: The formatted string based on the mission's requirements.
            """
            values = list(dictionary.values())
                    
            if input_mission == "mex":
                data, purpose, start_date_file, end_date_file, version, extension, start_date_utc, end_date_utc = values
                
                filename = f'{data}_{purpose}_{start_date_file}_{end_date_file}_{version}{extension}'
                return filename  # Using underscores

            elif input_mission == "mro":
                mission, instrument, purpose, start_date_file, end_date_file, extension, start_date_utc, end_date_utc = values
                filename = f'{mission}_{instrument}_{purpose}_{start_date_file}_{end_date_file}{extension}'
                return filename
            # Add more missions with specific formatting rules as needed
            else:
                return "_".join(expected_keys)  # Default behavior
            
    ########################### SECTION END: ORIENTATION CLASS ######################################