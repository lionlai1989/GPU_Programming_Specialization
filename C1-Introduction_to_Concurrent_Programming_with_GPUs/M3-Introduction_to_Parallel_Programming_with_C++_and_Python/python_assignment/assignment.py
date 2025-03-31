import argparse
import logging
import random
import sys
import time
from threading import Condition, Thread


def parse_arguments():
    parser = argparse.ArgumentParser(description="Threaded Ticketing System Assignment")
    parser.add_argument(
        "-n",
        "--num_threads",
        type=int,
        default=1,
        help="Number of concurrent threads to execute",
    )
    parser.add_argument(
        "-u",
        "--user",
        required=True,
        help="The user submitting the assignment, must match .user file contents",
    )
    parser.add_argument(
        "-p", "--part_id", default="test", help="The assignment ID, default is 'test'"
    )
    return parser.parse_args()


def execute_ticketing_system_participation(ticket_number, part_id, assignment_obj):
    output_file_name = "output-" + part_id + ".txt"
    # NOTE: Do not remove this print statement as it is used to grade assignment
    print(
        "Thread retrieved ticket number: {} started".format(ticket_number),
        file=open(output_file_name, "a"),
    )
    time.sleep(random.randint(0, 10))

    # Wait until the shared current_ticket equals the thread's ticket number
    with assignment_obj.condition:
        while ticket_number != assignment_obj.current_ticket:
            assignment_obj.condition.wait()

    # NOTE: Do not remove this print statement as it is used to grade assignment
    print(
        "Thread with ticket number: {} completed".format(ticket_number),
        file=open(output_file_name, "a"),
    )

    # Signal that this thread has finished by updating completed_ticket
    with assignment_obj.condition:
        assignment_obj.completed_ticket = ticket_number
        assignment_obj.condition.notify_all()
    return 0


class Assignment:
    USERNAME = "Lion"
    active_threads = []

    def __init__(self, args):
        self.file_username = None
        self.num_threads = args.num_threads
        self.username_arg = args.user
        self.part_id = args.part_id
        self.read_user_file()
        self.current_ticket = 0  # shared ticket counter; starts at 0
        self.completed_ticket = (
            -1
        )  # last completed ticket; initially -1 (none finished)
        self.condition = Condition()  # condition variable for synchronization

        logging.basicConfig(
            format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S"
        )

    def read_user_file(self):
        with open(".user", "r") as file:
            self.file_username = file.readline().strip()

    def test_username_equality(self, const_username):
        return self.file_username == self.username_arg == const_username

    def manage_ticketing_system(self):
        # For each thread, wait until the thread with the current ticket has signaled completion,
        # then increment the ticket counter.
        for _ in range(self.num_threads):
            with self.condition:
                # Wait until the thread corresponding to current_ticket has completed.
                while self.completed_ticket != self.current_ticket:
                    self.condition.wait()
                # Now that the thread with the current ticket is done, move to the next ticket.
                self.current_ticket += 1
                self.condition.notify_all()  # Wake up any waiting threads.
        return 0

    def run(self):
        output_file_name = f"output-{self.part_id}.txt"
        open(output_file_name, "w").close()
        if self.test_username_equality(self.USERNAME):
            threads = []

            # Create and start threads.
            for index in range(self.num_threads):
                logging.info("Creating and starting thread %d.", index)
                thread = Thread(
                    target=execute_ticketing_system_participation,
                    args=(index, self.part_id, self),
                )
                thread.start()
                threads.append(thread)

            # Start the ticket manager thread.
            manager_thread = Thread(target=self.manage_ticketing_system)
            manager_thread.start()

            # Wait for all threads to finish.
            for thread in threads:
                thread.join()
            manager_thread.join()

            logging.info("All threads completed.")
            return 0
        else:
            logging.error("Username mismatch. Please check code and .user file.")
            return 1


if __name__ == "__main__":
    args = parse_arguments()
    assignment = Assignment(args)
    sys.exit(assignment.run())
