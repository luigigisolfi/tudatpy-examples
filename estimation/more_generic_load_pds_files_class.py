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
        self.RadioScience = self.RadioScience(self) 

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
        ONCE THE PATTERN IS DEFINED, this can be added to the supported ones via something like:
        
        object = LoadPDS()
        object.add_custom_mission_pattern('luigi', 'blabla.bc')
        
        """
        custom_dict = {}
        custom_dict[input_mission] = custom_pattern
        self.Orientation.supported_orientation_formats.update(custom_dict)
        
        return self.Orientation.supported_orientation_formats

    def is_date_in_intervals(self, date, intervals):

        """
        
        This functions checks whether a given (wanted) date 
        falls within the time interval specified in the filename
        
        """
        
        self.in_intervals = False
        for interval in intervals:
            if ( ( date.date() >= interval[0].date() ) and ( date.date() <= interval[1].date() ) ):
                self.in_intervals = True
        return (self.in_intervals)

    def dynamic_download_url_files_time_interval(self, input_mission, local_path, filename_format, start_date, end_date, url, time_interval_format): 
        
        """
        Downloads files within a specified time interval if they are not already present in the local path.
    
        Parameters:
        - input_mission: Name of the mission (string).
        - local_path: Directory path where files are stored (string).
        - filename_format: The format of the filename to match (string).
        - start_date: The start date of the time interval (datetime).
        - end_date: The end date of the time interval (datetime).
        - url: The base URL for downloading files (string).
        - time_interval_format: Format for time intervals (not used in this function).
    
        Returns:
        - self.relevant_files: List of relevant files that were found or downloaded.
        
        """
    
        # Prepare the date range for searching existing files
        all_dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
        
        # Retrieve all existing files that match the filename format
        existing_files = glob.glob(local_path + filename_format)

        if existing_files:
            print(f'The following files already exist in the folder: {existing_files} and will not be downloaded.')
        
        # Initialize lists to hold relevant intervals and files
        relevant_intervals = []
        self.relevant_files = []    

        # Iterate through the existing files
        for full_filename in existing_files:
            filename = full_filename.split("/")[1]  # Extract filename from full path
            
            # Parse the orientation information from the filename
            orientation_dict, orientation_underscores = self.Orientation.parse_orientation_filename(input_mission, filename)

            # Reconstruct the filename to download
            filename_to_download = self.Orientation.reconstruct_orientation_filename(orientation_dict, orientation_underscores)  
            # Extract start and end times from the orientation dictionary
            start_time = orientation_dict["start_date_utc"]
            end_time = orientation_dict["end_date_utc"]
    
            is_relevant = False  # Flag to check if file is relevant
            date_cnt = 0  # Counter for all_dates
            current_interval = (start_time, end_time)  # Current time interval
    
            # Check if the current interval overlaps with any dates
            while not is_relevant and date_cnt < len(all_dates):
                if self.is_date_in_intervals(all_dates[date_cnt], [current_interval]):
                    is_relevant = True  # Found a relevant file
                    relevant_intervals.append(current_interval)  # Add the interval to the list
                date_cnt += 1  # Increment date counter
    
        # Identify dates that do not have corresponding files
        dates_without_file = [date for date in all_dates if not self.is_date_in_intervals(date, relevant_intervals)]
    
        # If there are dates without files, proceed to download
        if dates_without_file: 
            reqs = requests.get(url)  # Get the content of the URL
            files_url_dict = {}  # Dictionary to hold files with their time intervals
    
            # Parse links from the HTML response
            for link in BeautifulSoup(reqs.text, 'html.parser').find_all('a'):
                full_link = link.get('href')  # Extract the href attribute
                if isinstance(full_link, str):
                   
                    # Check if the link is a full URL or a relative path
                    if ('/') in full_link:
                        # It's a full URL
                        filename = os.path.basename(full_link)  # Get the filename from the URL
                    else:
                        # It's a relative path
                        filename = full_link
                        
                    #print(filename)
                    #print(filename_format.split('*')[0])  # Print the starting pattern
                    #print(filename_format.split('*')[1])  # Print the ending pattern
                    
                    # Check if the filename matches the specified format
                    if filename.startswith(filename_format.split('*')[0]) and filename.endswith(filename_format.split('*')[1]):
                        try:
                            # Parse the orientation information from the filename
                            orientation_dict, orientation_underscores = self.Orientation.parse_orientation_filename(input_mission, filename)
                        except:
                            print(f'Could not download orientation file: {filename}. Likely Reason: Filename Does Not Match Any Supported Pattern.')  # Handle parsing errors
                            continue  # Skip to the next link
                    else:
                        continue  # Skip to the next link
                    
                    # Reconstruct the filename and get time intervals
                    filename_to_download = self.Orientation.reconstruct_orientation_filename(orientation_dict, orientation_underscores)
                    start_time = orientation_dict["start_date_utc"]
                    end_time = orientation_dict["end_date_utc"]   
                    
                    # Store the filename in the dictionary with its time interval
                    files_url_dict[(start_time, end_time)] = filename
    
            # Download files for dates without corresponding files
            for date in dates_without_file:
                if not self.is_date_in_intervals(date, relevant_intervals):
                    for new_interval in files_url_dict:
                        if self.is_date_in_intervals(date, [new_interval]):
                            print('Downloading', files_url_dict[new_interval])  # Print which file is being downloaded
                            urlretrieve(url + files_url_dict[new_interval], local_path + files_url_dict[new_interval])  # Download the file
                            self.relevant_files.append(local_path + files_url_dict[new_interval])  # Add the downloaded file to relevant files
                            relevant_intervals.append(new_interval)  # Add the new interval to relevant intervals
    
        print('...Done.')  # Indicate completion
        return self.relevant_files  # Return the list of relevant files
        

    def dynamic_download_url_files_single_time(self, input_mission, local_path, filename_format, start_date, end_date, url, time_format):
        
        """
        This function downloads files for a single time interval based on specified mission parameters and date range.
    
        Parameters:
        - input_mission: Name of the mission (string).
        - local_path: Directory path where files are stored (string).
        - filename_format: The format of the filename to match (string).
        - start_date: The start date of the time interval (datetime).
        - end_date: The end date of the time interval (datetime).
        - url: The base URL for downloading files (string).
        - time_format: Format for the specified date (string).
    
        Returns:
        - self.relevant_files: List of relevant files that were found or downloaded.
        
        """
    
        # Determine the file type
        if ".tab" in filename_format:
            file_type = "IFMS"
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
            print(f'The following files already exist in the folder: {existing_files} and will not be downloaded.')
            for full_filename in existing_files:
                filename = os.path.basename(full_filename)  # Get the filename from the full path
            
                # Generate the expected filename
                if file_type == "IFMS":
                    try:
                        RS_dict, RS_underscores = self.RadioScience.parse_RS_filename(input_mission, filename)

                        filename_to_download = self.RadioScience.reconstruct_RS_filename(RS_dict, RS_underscores)
                    except Exception as e:
                        print(f"Error parsing RS IFMS filename {filename}: {e}")
                        continue
                elif file_type == "ODF":
                    try:
                        RS_dict, RS_underscores = self.RadioScience.parse_RS_filename(input_mission, filename)
                        filename_to_download = self.RadioScience.reconstruct_RS_filename(RS_dict, RS_underscores)
                    except Exception as e:
                        print(f"Error parsing RS ODF filename {filename}: {e}")
                        continue
            
                print('Expected filename to download:', filename_to_download)
    
                # Check if the file already exists
                if filename in [os.path.basename(x) for x in existing_files]:
                    print('File Already Exists in the Folder ;)')
                else:
                    print('File not found in local folders, will download:', filename_to_download)
    
        # If there are no existing files, we need to download all required files
        try:
            reqs = requests.get(url)
            reqs.raise_for_status()  # Raise an error for bad responses
    
            files_url_dict = {}
    
            # Retrieve files from the server
            for link in BeautifulSoup(reqs.text, 'html.parser').find_all('a'):
                full_link = link.get('href')
                if isinstance(full_link, str):
            
                    # Check if the link is a full URL or a relative path
                    if ('/') in full_link: #on pds, two things happen when you click on a link 1) download of file starts (relative url) or 2) link (full url) opens. 
                        # Full URL
                        filename = os.path.basename(full_link)
                    else:
                        # Relative path
                        filename = full_link
                    #print(filename)
                    #print(filename_format.split('*')[0])
                    #print(filename_format.split('*')[1])
                    if filename.startswith(filename_format.split('*')[0]) and filename.endswith(filename_format.split('*')[1]):
                            
                        try:
                            RS_dict, RS_underscores = self.RadioScience.parse_RS_filename(input_mission, filename)
                            #print(orientation_dict)
                            filename_to_download = self.RadioScience.reconstruct_RS_filename(RS_dict, RS_underscores)
                            #print(filename_to_download)

                            if input_mission == 'mex':
                                files_url_dict[RS_dict["date_file"][:5]] = filename_to_download #MEX: YYJJJHHMM, we only care about YYJJJ
                            else:
                                files_url_dict[RS_dict["date_file"][:8]] = filename_to_download #THESE SHOULD BE FINE
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

                if date_string in files_url_dict:
                    download_file = files_url_dict[date_string]
                    full_download_url = os.path.join(url, download_file)
                    print('Downloading:', full_download_url)
    
                    try:
                        urlretrieve(full_download_url, os.path.join(local_path, download_file))
                        self.relevant_files.append(os.path.join(local_path, download_file))
                    except Exception as e:
                        print(f"Failed to download {full_download_url}: {e}")
    
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {url}: {e}")

        print('...Done.')
        return self.relevant_files

    
    ########################### SECTION END: COMMON FUNCTIONS FOR ALL TYPES OF FILES ######################################


    ########################### SECTION START: IFMS CLASS ######################################

    class RadioScience:
        def __init__(self, LoadPDS):
            self.supported_RS_formats = {       
                "mex":
                    r'^(?P<mission>[a-zA-Z0-9]+)_(?P<band>[a-zA-Z0-9]+)_(?P<date_file>[0-9]{9})_(?P<version>[0-9]{2})(?P<extension>\.tab$)',
                "mro":
                    r'^(?P<mission>mro)(?P<dataset>magr)(?P<date_file>\d{4}_\d{3}_\d{4})(?P<uplink>x|n)(?P<station>nn|mm|[0-9]{2})(?P<downlink>[123m])(?P<version>v\d+)(?P<extension>\.tnf|.odf)$'
            }


        def parse_RS_filename(self, input_mission, filename):
            """ 
            Parses the orientation filename into its components and tracks the position of underscores. 
            The underscore position is indexed based on the groups' placement.
            """
        
            # Initialize the orientation dictionary and underscore index list
            RS_dict = {}
            underscore_indices = []
        
            # Attempt to retrieve the pattern for the specified mission directly
            if input_mission in self.supported_RS_formats:
                pattern = self.supported_RS_formats[input_mission]
                match = re.match(pattern, filename)
            else:
                raise ValueError('Selected Mission Not Supported (yet!) Aborting ...') 
        
            if match:
                # Populate the orientation dictionary with matched groups
                RS_dict = match.groupdict()
        
                # Add underscore index tracking based on group positions
                last_pos = 0
                group_index = 0
                for key in RS_dict.keys():
                    current_pos = match.start(key)  # Get the start position of the matched group
                    if last_pos != current_pos:  # If there was an underscore before this group
                        underscore_indices.append(group_index)  # Track the index of the underscore
                    last_pos = match.end(key)  # Move to the end of the current matched group
                    group_index += 1  # Increment the group index
        
                # Process dates if present
                if "start_date_file" in RS_dict and "end_date_file" in RS_dict:
                    self.start_date_utc = LoadPDS.generic_strptime(LoadPDS, RS_dict["start_date_file"]) 
                    self.end_date_utc = LoadPDS.generic_strptime(LoadPDS, RS_dict["end_date_file"])  
                    RS_dict["start_date_utc"] = self.start_date_utc
                    RS_dict["end_date_utc"] = self.end_date_utc
        
                return RS_dict, underscore_indices
            else:
                raise ValueError('Filename does not match any of the supported Radio Science formats.')
                

        def reconstruct_RS_filename(self, RS_dict, underscore_indices):
            """ 
            Reconstructs the original filename based on the parsed groups and underscore indices.
            Args:
                orientation_dict (dict): Dictionary with the parsed components of the filename.
                underscore_indices (list): List of group indices where underscores should be placed.
            Returns:
                str: The reconstructed filename.
            """
            
            # Extract the keys (group names) from the dictionary in order
            group_values = list(RS_dict.values())
            
            # Initialize an empty list to build the filename
            reconstructed_filename = []
            
            # Iterate through the group values and append to the list
            for i, group in enumerate(group_values):
                if isinstance(group, str):
                    reconstructed_filename.append(group)  # Add the current group
                
                # Check if the current index is in the underscore_indices list
                if i + 1 in underscore_indices:  # Underscore comes *after* the current group (between i and i+1)
                    reconstructed_filename.append("_")
            
            # Join the list into a single string (the reconstructed filename)
            return "".join(reconstructed_filename)
            
            
    ########################### SECTION END: IFMS CLASS ######################################
                
    ########################### SECTION START: ORIENTATION CLASS ###################################### 
                
    class Orientation:
    
        def __init__(self, LoadPDS):
            self.supported_orientation_formats = {
                "juice": 
                "^(?P<mission>juice)_(?P<instrument>(sc|mga|sa|lpbooms|magboom|majis_scan|swi_scan))_(?P<data_type>(attc|attm|meas|cmmd|plan|ptr|crema_[A-Z]+))_(?P<start_date_file>\d{6})_(?P<end_date_file>\d{6})_(?P<sclk>(s|t|f)\d{6})_(?P<version>v\d{2})(?P<extension>\.bc)$",         
                "mro":
                     r"^(?P<mission>mro)_(?P<instrument>[a-z]+)(?:_(?P<purpose>[a-z]+))?_(?P<start_date_file>\d{6})_(?P<end_date_file>\d{6})(?P<extension>\.bc)$",   
                "lro":
                    r"^(lro)([a-z]+)?(?:_([a-z]+))?_(\d{6}|\d{7})_(\d{6}|\d{7})(?:_(s\d{6}))?(?:_(v\d+))?(\.bc)$",
                "cassini":
                    r"^(?:([a-zA-Z0-9]+)_)?(\d{5})_(\d{5})([a-z]{0,2})?(\.bc)$"   
            }

            
        def parse_orientation_filename(self, input_mission, filename):
            """ 
            Parses the orientation filename into its components and tracks the position of underscores. 
            The underscore position is indexed based on the groups' placement.
            """
        
            # Initialize the orientation dictionary and underscore index list
            orientation_dict = {}
            underscore_indices = []
        
            # Attempt to retrieve the pattern for the specified mission directly
            if input_mission in self.supported_orientation_formats:
                pattern = self.supported_orientation_formats[input_mission]
                match = re.match(pattern, filename)
            else:
                raise ValueError('Selected Mission Not Supported (yet!) Aborting ...') 
        
            if match:
                # Populate the orientation dictionary with matched groups
                orientation_dict = match.groupdict()
        
                # Add underscore index tracking based on group positions
                last_pos = 0
                group_index = 0
                for key in orientation_dict.keys():
                    current_pos = match.start(key)  # Get the start position of the matched group
                    if last_pos != current_pos:  # If there was an underscore before this group
                        underscore_indices.append(group_index)  # Track the index of the underscore
                    last_pos = match.end(key)  # Move to the end of the current matched group
                    group_index += 1  # Increment the group index
        
                # Process dates if present
                if "start_date_file" in orientation_dict and "end_date_file" in orientation_dict:
                    self.start_date_utc = LoadPDS.generic_strptime(LoadPDS, orientation_dict["start_date_file"]) 
                    self.end_date_utc = LoadPDS.generic_strptime(LoadPDS, orientation_dict["end_date_file"])  
                    orientation_dict["start_date_utc"] = self.start_date_utc
                    orientation_dict["end_date_utc"] = self.end_date_utc
        
                return orientation_dict, underscore_indices
            else:
                raise ValueError('Filename does not match any of the supported orientation formats.')

        def reconstruct_orientation_filename(self, orientation_dict, underscore_indices):
            """ 
            Reconstructs the original filename based on the parsed groups and underscore indices.
            Args:
                orientation_dict (dict): Dictionary with the parsed components of the filename.
                underscore_indices (list): List of group indices where underscores should be placed.
            Returns:
                str: The reconstructed filename.
            """
            # Extract the keys (group names) from the dictionary in order
            group_values = list(orientation_dict.values())
            
            # Initialize an empty list to build the filename
            reconstructed_filename = []
            
            # Iterate through the group values and append to the list
            for i, group in enumerate(group_values):
                if isinstance(group, str):
                    reconstructed_filename.append(group)  # Add the current group
                
                # Check if the current index is in the underscore_indices list
                if i + 1 in underscore_indices:  # Underscore comes *after* the current group (between i and i+1)
                    reconstructed_filename.append("_")
            
            # Join the list into a single string (the reconstructed filename)
            return "".join(reconstructed_filename)
                            
    ########################### SECTION END: ORIENTATION CLASS ######################################