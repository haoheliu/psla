
from site import execusercustomize
import subprocess
import os
import re
import time
from datetime import datetime
import numpy as np
import logging
import os


# Do not need to change
SEPARATOR = "&"
IDENTIFIER_BASHFILE = "\nAUTOGEN=1\n"
RSYNC_CONDOR = "rsync.submitfile"
SUBMIT_CONDOR="submit.submitfile"
IGNORE_FILE = ".rsyncignore"
MAX_RUN = 5
RSYNC_MAX_RETRY=5
MIN_ELAPSE_TIME_AS_SUCCESSFUL = 15 * 60

if(os.path.exists("/mnt")):
    RSYNC_CONDOR_TEMPLATE = "/user/HS502/hl01486/Projects/condor/mnt_transfer_job_template.submitfile"  # Change path when you change computer
else:
    RSYNC_CONDOR_TEMPLATE = "/user/HS502/hl01486/Projects/condor/transfer_job_template.submitfile"  # Change path when you change computer

# May need changes
RSYNC_PERFORM = {"DATA": False, "LOG": False, "PROJECT": True}  # Transfer data or not
RSYNC_OPTIONS = {
    "DATA": "-a -v --ignore-existing", # This tells rsync to skip updating files that already exist on the destination (this does not ignore existing directories, or nothing would get done). See also --existing.
    "LOG": "-a -v --ignore-existing", 
    "PROJECT": "-a -v",
}  # The options for transfering data

# After the code excecution is end
CONDOR_GET_LOG = True  # Get the log file back to the original directory


logging.basicConfig(
    filename="submit.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

class CondorHelper:
    def __init__(self, submit_file) -> None:
        logging.info("Initializing new helper.")
        submit_file = os.path.abspath(submit_file)
        self.submit_file_path = submit_file
        if not os.path.exists(self.submit_file_path):
            logging.error("Condor submit file not found: " + self.submit_file_path)
            raise ValueError("Condor submit file not found: " + self.submit_file_path)

        self.submit_sh_file_path = os.path.join(
            os.path.dirname(self.submit_file_path), "submit.sh"
        )
        if not os.path.exists(self.submit_sh_file_path):
            logging.error("submit bash file not found: " + self.submit_sh_file_path)
            raise ValueError("submit bash file not found: " + self.submit_sh_file_path)

        os.makedirs(os.path.join(os.path.dirname(self.submit_file_path),"condor_log"), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(self.submit_file_path),"condor_file_log"), exist_ok=True)

        self.log_src, self.log_dest = None, None
        self.data_src, self.data_dest = None, None
        self.project_src, self.project_dest = None, None

        self.identifier = self._get_identifier()
        self._parse_bash_file()

        if (
            self.project_src[0] not in self.log_src[0]
            or self.project_dest[0] not in self.log_dest[0]
        ):
            logging.error(
                "The log directory should in the project folder. %s %s %s %s"
                % (self.log_src, self.project_src, self.log_dest, self.project_dest)
            )
            raise ValueError(
                "The log directory should in the project folder. %s %s %s %s"
                % (self.log_src, self.project_src, self.log_dest, self.project_dest)
            )

    def submit(self):
        """
        1. Transfer data from src to dest. (If the data is there, rsync will auto skip.)
        2. Transfer project code from src to dest. Will not transfer log.
        3. (Optional) If specified, transfer log from src to dest. Default is False.
        4. After running finished. Transfer log from dest to src.
        5. End
        """
        if RSYNC_PERFORM["DATA"]:
            logging.info("Transfering data...")
            success, i = self._transfer_data(self.data_src, self.data_dest, RSYNC_OPTIONS["DATA"]), RSYNC_MAX_RETRY
            while(not success and i > 0):
                logging.error("Transfering data not success. No files appears in the destination. Retrying...")
                self._transfer_data()
                i-=1

        if RSYNC_PERFORM["PROJECT"]:
            logging.info("Transfering project...")
            success, i = self._transfer_project(), RSYNC_MAX_RETRY
            while(not success and i > 0):
                logging.error("Transfering project not success. No files appears in the destination. Retrying...")
                self._transfer_project()
                i-=1

        if RSYNC_PERFORM["LOG"]:
            logging.info("Transfering logs...")
            success, i = self._transfer_data(self.log_src, self.log_dest, RSYNC_OPTIONS["LOG"]), RSYNC_MAX_RETRY
            while(not success and i > 0):
                logging.error("Transfering log not success. No files appears in the destination. Retrying...")
                self._transfer_log()
                i-=1

        self.update_bash_file(self.submit_sh_file_path)
        submifile_path = self.update_submit_file(self.submit_file_path)

        i = MAX_RUN
        while i > 0:
            elapse_time, cluster_id = self.excute_condor_submit(
                submifile_path
            )
            if CONDOR_GET_LOG:
                logging.info("Transfering logs back...")
                success, i = self._transfer_data(self.log_dest, self.log_src, RSYNC_OPTIONS["LOG"]), RSYNC_MAX_RETRY
                while(not success and i > 0):
                    logging.error("Transfering log back not success. No files appears in the destination. Retrying...")
                    self._transfer_log_back()
                    i-=1

            # If the code is not succesfully executed
            if elapse_time < MIN_ELAPSE_TIME_AS_SUCCESSFUL:
                logging.error(
                    "The elapse time is only %s seconds. Something wrong happens."
                    % elapse_time
                )
                break
            # If the code is succesfully executed
            else:
                self.excute("condor_rm %s" % cluster_id)
                i -= 1

    def excute(self, command, verbose=True):
        """Excute a linux shell command

        Args:
            command (_type_): _description_

        Returns:
            _type_: _description_
        """
        if(verbose):
            logging.info(command)
        result = subprocess.run(command.split(" "), stdout=subprocess.PIPE)
        return result

    def excute_sys(self, command):
        os.system(command)

    def excute_condor_submit(self, submitfile_path, command="echo \"test\""):
        """Submit condor submitfile. Will wait until the excution is finished.
        Args:
            submitfile_path (_type_): _description_
        """

        def submit(submitfile_path):
            msg = self.excute("condor_submit %s" % submitfile_path)
            stdout = str(msg.stdout).split("submitted to cluster")[-1]
            cluster_id = stdout.split(".")[0].strip()
            return cluster_id

        def check_if_finish(cluster_id):
            msg = self.excute("condor_q %s" % cluster_id, verbose=False)
            msg = str(msg.stdout).strip("b\\n\'")
            idle = int(re.findall("\d+ idle", msg)[0].split(" ")[0])
            running = int(re.findall("\d+ running", msg)[0].split(" ")[0])
            if(running==0 and idle==0):
                return True
            else:
                return False

        cluster_id = submit(submitfile_path)
        elapse_time = 0
        logging.info("Submit successful! Running on %s" % cluster_id)
        while not check_if_finish(cluster_id):
            time.sleep(1)
            elapse_time += 1
            if elapse_time % 3 == 0:
                logging.debug(
                    "Waiting for %s to finish, elapse time %s minutes %s second"
                    % (submitfile_path, elapse_time // 60, elapse_time % 60)
                )
        logging.info(
            "%s excution finished, time cost %s minutes %s seconds"
            % (submitfile_path, elapse_time // 60, elapse_time % 60)
        )
        return elapse_time, cluster_id

    def _parse_bash_file(self):
        bash_file = self._read_file_to_str(self.submit_sh_file_path)
        # The variable that store the data path we need to transfer
        self.data_src, self.data_dest = self._parse_variable(bash_file, "DATA")
        self.log_src, self.log_dest = self._parse_variable(bash_file, "LOG")
        self.project_src, self.project_dest = self._parse_variable(bash_file, "PROJECT")
        assert (
            len(self.log_src) == 1
            and len(self.log_dest) == 1
            and len(self.project_src) == 1
            and len(self.project_dest) == 1
        )
        # Update the submitfile path from val to mnt
        self.submit_file_path = self.submit_file_path.replace(
            self.project_src[0], self.project_dest[0]
        )
        self.submit_sh_file_path = self.submit_sh_file_path.replace(
            self.project_src[0], self.project_dest[0]
        )
        
        os.makedirs(os.path.join(os.path.dirname(self.submit_file_path),"condor_log"), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(self.submit_file_path),"condor_file_log"), exist_ok=True)

    def add_slash(self, str):
        # rsync interprets a directory with no trailing slash as copy this directory, and a directory with a trailing slash as copy the contents of this directory.
        if str[-1] != "/":
            return str + "/"
        else:
            return str

    def _transfer_data(self, srcs, dests, options):
        for src, dest in zip(srcs, dests):
            src, dest = self.add_slash(src), self.add_slash(dest)
            os.makedirs(dest, exist_ok=True)
            if not os.path.exists(src):
                logging.error("The rsync source path %s is not exist." % src)
                continue
            rsync_condor, command = self._rsync_condor_submit_file(
                options, src, dest, RSYNC_CONDOR
            )
            self.excute_condor_submit(rsync_condor, command)
            if(len(os.listdir(src)) > 1 and len(os.listdir(dest)) < 1): return False
        return True

    def _transfer_project(self):
        for src, dest in zip(self.project_src, self.project_dest):
            src, dest = self.add_slash(src), self.add_slash(dest)
            os.makedirs(dest, exist_ok=True)
            if not os.path.exists(src):
                logging.error("The rsync source path %s is not exist." % src)
                continue
            if not os.path.exists(os.path.join(self.project_src[0], IGNORE_FILE)):
                msg = "The .rsyncignore file is not found in the project folder."
                logging.error(msg); raise ValueError(msg)
            exclude_paths = self._read_file_to_list(os.path.join(self.project_src[0], IGNORE_FILE))
            options = RSYNC_OPTIONS["PROJECT"] 
            for path in exclude_paths:
                options += " --exclude=%s " % os.path.join(self.project_src[0], path)
            rsync_condor, command = self._rsync_condor_submit_file(
                options,
                src,
                dest,
                RSYNC_CONDOR,
            )
            self.excute_condor_submit(rsync_condor, command)
            if(len(os.listdir(src)) > 1 and len(os.listdir(dest)) < 1): return False
        return True
        
    def _rsync_condor_submit_file(self, options, src, dest, fname=RSYNC_CONDOR):
        template = self._read_file_raw(RSYNC_CONDOR_TEMPLATE)
        if("$OPTIONS" not in template or "$SRC" not in template or "$DEST" not in template):
            msg = "The template need to contain DEST, SRC, and OPTIONS."
            logging.error(msg); raise ValueError(msg)
        template = (
            template.replace("$OPTIONS", options)
            .replace("$SRC", src)
            .replace("$DEST", dest)
        )
        self._write_file_raw(template, fname)
        command = "rsync %s %s %s" % (options, src, dest)
        logging.info("Prepare to exec: " + command)
        return fname, command

    def _get_identifier(self):
        now = datetime.now()
        return now.strftime("%d_%m_%Y_%H_%M")

    def _parse_variable(self, content, variable):
        varibale = re.findall(r'%s="\S+"' % (variable), content)
        if len(varibale) != 1:
            logging.error("Error: invalid definition of the variable: " + str(varibale))
            raise ValueError(
                "Error: invalid definition of the variable: " + str(varibale)
            )
        # Get the variable content between two ""
        variable = varibale[0].split('="')[1][:-1]
        variable = variable.replace("$IDENTIFIER", self.identifier)
        src, dest = [], []
        for var in variable.split(","):
            x, y = var.split(SEPARATOR)
            src.append(x.strip(" "))
            dest.append(y.strip(" "))
        return src, dest

    def _read_file_to_list(self, filepath):
        with open(filepath, "r") as f:
            data = f.readlines()
        return [x.strip("\n") for x in data]

    def _read_file_to_str(self, filepath):
        with open(filepath, "r") as f:
            data = f.read().replace("\n", " ")
        return data

    def _read_file_raw(self, filepath):
        with open(filepath, "r") as f:
            data = f.read()
        return data

    def _write_file_raw(self, content, filepath):
        with open(filepath, "w") as f:
            f.write(content)

    def update_bash_file(self, bashfile_path):
        # Append some new content onto the submit file
        ## Rsync the log file to the original log folder
        file = self._read_file_raw(bashfile_path)
        # Clear the command added by previous condor job
        if IDENTIFIER_BASHFILE in file:
            file = file.split(IDENTIFIER_BASHFILE)[1]
        file = IDENTIFIER_BASHFILE + file
        file = "IDENTIFIER=%s\n" % self.identifier + file
        self._write_file_raw(file, bashfile_path)

    def update_submit_file(self, submitfile_path):
        file = self._read_file_raw(submitfile_path)
        ## Make a new submitfile. And update the bash script path.
        if("$SH_FILE" not in file):
            msg = "You need to put $SH_FILE in the submit file target path."
            logging.error(msg); raise ValueError(msg)
        file = file.replace("$SH_FILE", self.submit_sh_file_path+" ")
        self._write_file_raw(file, SUBMIT_CONDOR)
        return SUBMIT_CONDOR

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-f', type=str)

    args = parser.parse_args()

    if(not os.path.exists(args.f)):
        raise ValueError("The submit file is not exist.")

    logging.info("===========================================")
    helper = CondorHelper(args.f)
    try:
        helper.submit()
    except Exception:
        logging.exception("message") 
