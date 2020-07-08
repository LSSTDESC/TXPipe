import paramiko
import warnings
import time


class NerscMonitor:
    def __init__(
        self, dirnames, username=None, key_filename=None, client=None, interval=3
    ):
        self.dirnames = dirnames
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if client is None:
                client = paramiko.SSHClient()
                client.load_system_host_keys()
                client.connect(
                    'cori.nersc.gov', username=username, key_filename=key_filename
                )
            self.client = client
        self._stdouts = None
        self._stderrs = None
        self.interval = interval
        self._time_of_last_monitor = -9999999

    def time_since_last_monitor(self):
        return time.time() - self._time_of_last_monitor

    def _start_check(self):
        # Don't start things too often
        if self.time_since_last_monitor() < self.interval:
            return

        outs = []
        errs = []
        for dirname in self.dirnames:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, out, err = self.client.exec_command('ls -1 ' + dirname)
            outs.append(out)
            errs.append(err)
        self._stdouts = outs
        self._stderrs = errs
        self._time_of_last_monitor = time.time()

    def _parse_results(self):
        files = []

        for dirname, stdout, stderr in zip(self.dirnames, self._stdouts, self._stderrs):
            err = stderr.read()
            if err:
                raise IOError(
                    f"Error connecting or listing remote dir {dirname}:\n{err}"
                )
            files_1 = [
                f.strip()
                for f in stdout.read().decode('utf-8').split("\n")
                if f.strip()
            ]
            files.append(files_1)

        # reset markers
        self._stdouts = None
        self._stderrs = None
        return files

    def _results_ready(self):
        return all(stdout.channel.exit_status_ready() for stdout in self._stdouts)

    def update(self):
        if self._stdouts is None:
            self._start_check()
            return None
        elif self._results_ready():
            return self._parse_results()
        else:
            return None
