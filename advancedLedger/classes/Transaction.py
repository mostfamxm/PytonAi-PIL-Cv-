import uuid


def power_of_x(number, x):
    return number ** x


class Transaction:
    ledger = []  # class variable that maintains the transactions created

    # constructor method
    def __init__(self, amount, acc_number):
        self.account = acc_number
        self.amount = amount
        self.trans_id = str(uuid.uuid4())
        self.ledger = []  # instance member with the ledger
        Transaction.ledger.append(self.trans_id)

    # instance method
    def get_class_name(self):
        self.instance_member = "created in get_class_name method"
        return str(self.__class__)  # in Java: this.getClass() Class<?>

    def add_transaction(self, another_transaction):
        """
        This method adds the ids to the ledgers of both transactions
        :param another_transaction:
        """
        if isinstance(another_transaction, Transaction):
            self.ledger.append(another_transaction.trans_id)
            another_transaction.ledger.append(self.trans_id)
        else:
            raise ValueError("another_transaction is of type ", another_transaction.__class__)