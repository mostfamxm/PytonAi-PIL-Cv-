import datetime
import hashlib


class Blockchain:

    def __init__(self):
        self.chain = []
        self.create_block(proof=1, previous_hash='0')
        self.current_transactions = []

    def create_block(self, proof, previous_hash):
        block = {'index': len(self.chain) + 1,
                 'timestamp': str(datetime.datetime.now()),
                 'proof': proof,
                 'previous_hash': previous_hash}
        self.chain.append(block)

        return block

    def last_block(self):
        return self.chain[-1]

    def add_block(self, block, proof):
        previous_hash = self.last_block.hash
        if previous_hash != block.previous_hash:
            return False
        if not self.is_valid_proof(block, proof):
            return False
        block.hash = proof
        self.chain.append(block)
        return True

    def add_transaction_to_queue(self, trans_obj):
        """
        adds new transaction to the transactions queue
        :param trans_obj: object of type transaction

        """
        self.current_transactions.append(trans_obj)

        return int(self.last_block['index']) + 1
