from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import re
from EmailBase import EmailBase
#from email_paths import ExtractEmailAddress,IdentifyEmailAddress




class EmailCounts(EmailBase):
    def __init__(self,email_paths, reference_emails, dataset):
        self.email_paths = email_paths
        self.reference_emails = reference_emails
        self.dataset = dataset
        self.dict_count = defaultdict(int)
        self.dict_timestamp = defaultdict(list)
        self.unique_emails = list()
    
    

    def processEmployee(self,email_feature):
        """
        for each employee present in the main dataset, initiates processing of emails that are authored by the
        respective employee.
        
        email_feature: argument defining the context for analysis of emails (string)
        -valid entries: sent_messages, received_messages, sent_to_poi, received_from_poi
        """
            
        for employee in self.email_paths.keys():
            # ignore emails from unknown authors for all types of email features but for "received_messages" and "shared messages"
            # --> unknown authors only releveant when calculating total number of received messages or shared receipt with poi
            if email_feature != "received_messages" and email_feature != "shared":
                if employee == "unknown employee":
                    continue

            self.processEmployeeEmails(employee,email_feature)


    def processEmployeeEmails(self,employee,email_feature):
        """
        extracts email metadata required for analysis of a single email and initiates analysis
        
        email_feature: see processEmployee method
        employee: email author (name of enron employee; string)
        """
            
        for path in self.email_paths[employee]:
            email = open(path, "r")
            email_text = email.read()
            
            # get author email address
            from_email = self.ExtractEmailAddress(email_text, identifier="sent")
            # get recepient email address
            to_email = self.ExtractEmailAddress(email_text, identifier="received")
            # get shared email address
            cc_email = self.ExtractEmailAddress(email_text, identifier="shared")
            # get name of email author (returns False if email author not an enron employee)
            author = self.IdentifyEmailAddress(from_email[0], self.reference_emails)
            # check if email author is poi
            is_poi_author = self.ClassAffiliation(author, self.dataset)
            # get email timestamp
            timestamp = self.EmailTimestamp(email_text)
            # get custom email ID
            email_id = self.EmailID(email_text)
            # close email object
            email.close()
            
            email_set = self.defineEmailSet(cc_email,to_email,email_feature)
            self.getEmailAddress(email_set,email_id, employee,is_poi_author,timestamp,email_feature)
        
        
        
    def defineEmailSet(self,cc_email,to_email,email_feature):
        """
        defines and returns set of recipient email addresses to be used in the context of analysis
        email_feature: see processEmployee method
        to_email: list of email addresses directly assigned as recipients
        cc_email: list of email addresses specified as carbon copy recipients
        """
        
        # combine to and cc emails to a set if analysing shared receipt with poi; if no cc emails, only
        # to emails will be used.
        if email_feature == "shared" and cc_email and to_email:
            email_set = set(to_email + cc_email)
        else:
            email_set = to_email
        return email_set
                
                
                
    def getEmailAddress(self,email_set,email_id, employee,is_poi_author,timestamp,email_feature):
        """
            Checks if email is unique and iterates over all recipient email addresses and forwards each
            to the analysis pipeline.
            
            email_set: set of all recipient email addresses (set)
            email_id: unique email identifier (see EmailID funstion in email_processing.py)
            employee: email author (name of enron employee; string)
            is_poi_author: is email author a person of interest (boolean)
            timestamp: email timestamp (string)
            email_feature: see processEmployee method
            """

        # check if email has any recepient and is not a duplicate
        if email_set and email_id not in self.unique_emails:
            self.unique_emails.append(email_id)
            tmp_dict = dict()
            unique_recipients = set()
            
            # use set because in some emails the same recipient address is used repeatedly!!
            for idx, address in enumerate(set(email_set)):
                self.analyseEmail(employee,address,unique_recipients,tmp_dict,idx,timestamp,is_poi_author,email_feature)
            
            self.updateEmailStatus(employee,tmp_dict,email_feature)



    def analyseEmail(self,employee,address,unique_recipients,tmp_dict,idx,timestamp,is_poi_author,email_feature):
        """
        evaluates the context of the email (provided via email_feature argument) and increases the counter by 1,
        i.e if the total number of emails sent by an employee is supposed to be recorded, the analysis pipeline 
        will increase the appropriate counter (for each employee) by 1 for each recipient in all emails.
            
        employee: email author (name of enron employee; string)
        address: email address (used to identify
        """
        
        # get name of email recepient
        # --> count addresses in email set to discriminate multiple unknown addresses
        recipient = self.IdentifyEmailAddress(address, self.reference_emails)
        if recipient == "unknown employee":
            recipient = "{0}_{1}".format(recipient,idx)
    
        # count same recipient with multiple email addresses only once
        if recipient not in unique_recipients:
            unique_recipients.add(recipient)
            
            # check if recepient is poi
            is_poi = self.ClassAffiliation(recipient, self.dataset)
            
            # ignore email if author is also the recipient
            if recipient and recipient != employee:
                # register all recipients, also unknown
                tmp_dict[recipient] = is_poi
                
                if email_feature == "sent_messages":
                    self.dict_count[employee] += 1
                    self.EmailSpecification(employee, timestamp, is_poi)
                # exclude unknown recipients from received messages count (only employees are relevant)
                if email_feature == "received_messages" and "unknown employee" not in recipient:
                    self.dict_count[recipient] += 1
                    self.EmailSpecification(recipient, timestamp, is_poi_author)
        
            if email_feature == "sent_to_poi":
                if is_poi and recipient != employee:
                    self.dict_count[employee] += 1
            # exclude unknown recipients from received messages count (only employees are relevant)
            if email_feature == "received_from_poi" and "unknown employee" not in recipient:
                if is_poi_author and recipient and recipient != employee:
                    self.dict_count[recipient] += 1



    def EmailSpecification(self, source, email_timestamp, poi_status):
        '''
        For each sent or received email, email timestamp and specification about poi status of author or
        recipient is
        appended to the timestamp_dict value (defaultdict initiated as a list). Appended entries are lists
        themselves.
        --> Format of timestamp_dict: {"name":[[email_timestamp], [author/recipient_specification]]}
        
        source: email author or email recipient (string)
        email_timestamp: timestamp of processed email (string)
        poi_status: indicator of poi status for email author or recipient (bool)
        '''
        # add lists with timestamp and poi_status
        if source not in self.dict_timestamp.keys():
            self.dict_timestamp[source].append([email_timestamp])
            self.dict_timestamp[source].append([poi_status])
                
        # update timestamp and poi_status lists
        elif source in self.dict_timestamp.keys():
            for idx,arg in enumerate([email_timestamp, poi_status]):
                self.dict_timestamp[source][idx].append(arg)



    def updateEmailStatus(self,employee,tmp_dict,email_feature):
        """
        Checks if email was sent exclusively to person of interest-type of recipients and updates
        dict_timestamp accordingly (sets True if all recipients are person of interes, otherwise False).
        --> required for 'difference in frequency of sent/received emails' feature engineering
            
        Alternatively, checks for email recipients if email was shared with a person of interest (specified by
        'shared' as email_feature argument)
            
        employee: email author (name of enron employee; string)
        tmp_dict: storage of email recipients and the respective poi-status (for each email;dictionary)
        email_feature: see processEmployee method
        """
    
        # update timestamp dictionary for received emails
        #--> append True if all recipients are poi, else append False
        if email_feature == "sent_messages": #update for employee sending emails
            for recipient_name in tmp_dict.keys():
                self.PoiExclusiveEmail(employee,tmp_dict)
        
        if email_feature == "received_messages": #update for employee receiving emails
            for recipient_name in tmp_dict.keys():
                # ignore unknown recipients; excluded from received messages count
                if "unknown employee" not in recipient_name:
                    self.PoiExclusiveEmail(recipient_name,tmp_dict)
    
        if email_feature == "shared":
            for name in tmp_dict.keys():
                tmp_pop = tmp_dict.pop(name)
                if any(tmp_dict.values()) and "unknown employee" not in name:
                    self.dict_count[name] += 1
                tmp_dict[name] = tmp_pop


    def PoiExclusiveEmail(self,source,tmp_dict):
        '''
        appends bool (as a list) to the timestamp_dict value indicating whether email recipent set consists of
        a poi only group (either single recipient or multiple recipients)
        --> Format of timestamp_dict: {"name":[[email_timestamp], [author/recipient_specification], [recipient_set-specification]]}
        
        source: email author or email recipient (string)
        '''
            # appends empty list as placeholder for new item
        if len(self.dict_timestamp[source]) != 3:
            self.dict_timestamp[source].append([])
        if not all(tmp_dict.values()):
            self.dict_timestamp[source][2].append(False)
        else:
            self.dict_timestamp[source][2].append(True)


    def updateMissingEmployee(self,counts_dict):
        """
        For each employee from the main dataset with no email information available, counter is set to NaN
        """
        # add employee with zero count if no email sent or received
        for name in self.dataset.index.values:
            #if name not in self.dict_count.keys():
            if name not in counts_dict.keys():
                #self.dict_count[name] = np.nan
                counts_dict[name] = np.nan


    def clear(self):
        """
        Reinitiates dict_count,dict_timestamp and unique_emails object variables (allows to reinitiate the
        analysis for different email contexts using the same instance of the EmailCounts class).
        """
        self.dict_count = defaultdict(int)
        self.dict_timestamp = defaultdict(list)
        self.unique_emails = list()
    
    
    def fit(self,email_feature):
        """
        initiates email analysis and returns a dictionary with appropriate counts (specified via email_feature 
        argument) for each employee from the main dataset;
        if appropriate returns a dictionary with email timestamps and author/recipient-set poi status
        (see PoiExclusiveEmail method)
            
        email_feature: see processEmployee method
        """
        self.clear()
        self.processEmployee(email_feature)
        self.updateMissingEmployee(self.dict_count)
        
        if email_feature == "sent_messages" or email_feature == "received_messages":
            return self.dict_count, self.dict_timestamp
        return self.dict_count




class EmailFrequencyChange(EmailCounts):
    '''
    returns the difference in counts of emails sent or received with respect to a reference date,
    i.e. count after reference date - count before reference date.
        
    email_timestamp: timestamp of processed email (string)
    dataset: main dataset (dataframe)
    '''
    def __init__(self,dataset):
        #self.email_timestamp = email_timestamp
        self.dataset = dataset
        self.reference_timestamp = datetime(2001, 8,15)
        self.count_diff = dict()
    
    
    def processEmployee(self,email_timestamp):
        
        for employee in email_timestamp.keys():
            # initiate counts for emails sent/received before (0) and after (1) refernce date
            count_all = [0,0]
            count_poi = [0,0]
            self.processEmails(employee,count_all,count_poi,email_timestamp)


    def processEmails(self,employee,count_all,count_poi,email_timestamp):
    
        # check if employee has any sent/received emails
        if len(email_timestamp[employee]) > 0:
            for idx,timestamp in enumerate(email_timestamp[employee][0]):
            
                delta =  timestamp < self.reference_timestamp # if email before reference date
                increment = 1
                poi_author_recipient = email_timestamp[employee][1][idx]  # bool; poi status of author/recipient
                poi_only_recipients = email_timestamp[employee][2][idx]   # bool; is recipient set poi only
                
                if poi_only_recipients:
                    increment = 10
                if delta:
                    if poi_author_recipient:
                        count_poi[0] += increment
                    count_all[0] += increment
                else:
                    if poi_author_recipient:
                        count_poi[1] += increment
                    count_all[1] += increment

            self.count_diff[employee] = sum(count_poi)/float(sum(count_all))
            self.normalizeFrequency(employee,count_all,count_poi)


    def normalizeFrequency(self,employee,count_all,count_poi):
    
        # return no change in frequency if no email communication at all or if no email communication to/from poi
        if sum(count_all) == 0 or sum(count_poi) == 0:
            self.count_diff[employee] = 1
    
        else:
            # normalize to total email count; upscale lower count
            lower_count = np.argmin(count_all)
            higher_count = np.argmax(count_all)
            
            if min(count_all) == 0:
                normalizing_factor = max(count_all)
            else:
                normalizing_factor = (count_all[higher_count]/float(count_all[lower_count]))
    
            if count_poi[lower_count] == 0:
                count_poi[lower_count] = normalizing_factor
            else:
                count_poi[lower_count] = normalizing_factor*count_poi[lower_count]
        
            # calculate normalized frequency change in email communication to/from poi
            try:
                self.count_diff[employee] = round(count_poi[1] / float(count_poi[0]),2)
            except ZeroDivisionError:
                self.count_diff[employee] = round(count_poi[1],2)
                    
                
                    
    def fit(self,email_timestamp):
        self.processEmployee(email_timestamp)
        self.updateMissingEmployee(self.count_diff)
        return self.count_diff
