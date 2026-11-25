import math

class PriorityQueue:
    def __init__(self,  f):
        self.queue = []
        self.f = f

    def append(self, item, hi=None):
        """Insert item at its correct position."""

        # Search for insertion index taken from https://github.com/python/cpython/blob/3.6/Lib/bisect.py
        lo = 0
        hi = hi

        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(self.queue)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.f(self.queue[mid], item):
                lo = mid + 1
            else:
                hi = mid

        self.queue.insert(lo, item)

    def extend(self, items):
        """Insert each item in items at its correct position."""
        for item in items:
            self.append(item)

    def pop(self):
        """Pop and return the first item in queue."""
        if self.queue:
            return self.queue.pop(0)
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    def remove(self, item):
        """Removes the item from queue."""
        self.queue.remove(item)

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return len(self.queue)

    def __contains__(self, item):
        """Return True if the key is in PriorityQueue."""
        return any([ite == item for ite in self.queue])


class PriorityQueueLex:
    def __init__(self, nof_rules, lower_bound_value, violation_fct, comparison_ftc, evaluation_fct=None):
        self.nof_rules = nof_rules
        self.buckets = [set() for i in range(nof_rules)]
        self.lower_bound_value = lower_bound_value

        self.violation_fct = violation_fct
        self.comparison_ftc = comparison_ftc
        self.evaluation_fct = evaluation_fct

    def append(self, item):
        bucket_idx = self.comparison_ftc(self.lower_bound_value, item, self.violation_fct)
        self.buckets[bucket_idx].add(item)

    def pop(self):
        # Find node with the lowest violation, set self.lower_bound_violation, remove node from bucket and return node
        for bucket_idx, bucket in reversed(list(enumerate(self.buckets))):
            if len(bucket) > 0:
                lower_bound_item = self.findLowerBoundItemAndRemove(bucket_idx)
                self.lower_bound_value = self.violation_fct(lower_bound_item)
                return lower_bound_item
        return None

    def findLowerBoundItemAndRemove(self, bucket_idx):
        # Find lower bound item in lowest non-empty bucket and assign remaining items to respective bucket
        bucket = self.buckets[bucket_idx]
        if self.evaluation_fct is not None: # If we first have to evaluate the items on the respective level
            for item in bucket: self.evaluation_fct(item, bucket_idx)
        min_item = min(bucket, key=lambda n: self.violation_fct(n)[bucket_idx]) #Todo This is probably super inefficient -> Can we directly access the violation?

        #If current bucket is the lowest bucket
        if bucket_idx == self.nof_rules -1:
            bucket.remove(min_item)
            return min_item
        else:
            min_items = {item for item in bucket if math.isclose(self.violation_fct(item)[bucket_idx], self.violation_fct(min_item)[bucket_idx], abs_tol=1e-9)}
            bucket -= min_items

            if len(min_items) == 1:
                return min_items.pop()
            else:
                self.buckets[bucket_idx+1] = min_items
                return self.findLowerBoundItemAndRemove(bucket_idx+1)


    def remove(self, item):
        # Remove item from corresponding bucket
        for bucket in self.buckets:
            if item in bucket:
                bucket.remove(item)
                break

    def __contains__(self, item):
        """Return True if the item is in Priority Queue."""
        return any([(item in bucket) for bucket in self.buckets])

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return sum([len(bucket) for bucket in self.buckets])
