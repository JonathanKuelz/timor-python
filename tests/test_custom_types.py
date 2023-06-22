from dataclasses import dataclass
import datetime
import random
import unittest

import numpy as np

from timor.utilities.dtypes import EternalDict, SingleSet, TypedHeader
import timor.utilities.errors as err


class CustomTypeUnitTests(unittest.TestCase):
    """Tests dtypes in utilities."""

    def setUp(self) -> None:
        # Fix random seeds to have deterministic tests
        random.seed(123)
        np.random.seed(123)

    def test_EternalDict(self):
        d = EternalDict()
        d['first'] = 1
        with self.assertRaises(TypeError):
            d['first'] = 2
        d[2] = 2
        with self.assertRaises(TypeError):
            d[2.] = 2  # lol, python
        d_1 = EternalDict(name='bond', long_name='james bond')
        d_2 = EternalDict(name='bond', profession='agent')
        with self.assertRaises(TypeError):
            d_3 = d_1.update(d_2)
        with self.assertRaises(TypeError):
            del d[2]

    def test_SingleSet(self):
        robots = SingleSet()
        robots.add('Panda')
        robots.update(['R2D2', 'Marvin', 'ModRob'])

        names = SingleSet(['Dimitrij', 'Svetlana', 'Marvin'])
        places = SingleSet(['NYC', 'Garching Forschungszentrum', 'Tokyo'])

        original_robots = robots.copy()
        with self.assertRaises(err.UniqueValueError):
            robots.update(names)  # The name "Marvin" is contained in robots and names
        with self.assertRaises(err.UniqueValueError):
            robots.union(names)
        self.assertEqual(robots, original_robots)  # If update fails, it must fail completely

        self.assertIsNone(robots.update(places))  # This however must work
        self.assertEqual(robots, set(robots).union(places))
        self.assertEqual(set(robots), set(robots).union(places))

        self.assertIsInstance(original_robots.union(places), SingleSet)  # Check if union returns the right dtype

        # Finally, a test with random numbers:
        for _ in range(100):
            numbers = np.random.randint(0, 20, (20,))
            original_set = set(numbers)
            if len(numbers) == len(original_set):  # No duplicates
                self.assertEqual(original_set, SingleSet(numbers))
                original_set.pop()
                self.assertNotEqual(original_set, SingleSet(numbers))
            else:
                with self.assertRaises(err.UniqueValueError):
                    SingleSet(numbers)

    def test_StronglyTypedHeader(self):
        @dataclass
        class TestHeader(TypedHeader):
            name: str
            number: int = 2
            date: datetime.date = datetime.date.today()

        header = TestHeader(name='test')
        with self.assertRaises(TypeError):
            header.name = 'new_name'
        with self.assertRaises(TypeError):
            header.__delattr__('name')

        header = TestHeader(name=666, date=datetime.date(1970, 1, 1))
        self.assertIsInstance(header.name, str)
        self.assertEqual(header.name, '666')

        first_date = TestHeader(name='test', date='1970-01-01')
        second_date = TestHeader(name='test', date='1970/1/1')
        self.assertEqual(header.date, first_date.date)
        self.assertEqual(header.date, second_date.date)


if __name__ == '__main__':
    unittest.main()
