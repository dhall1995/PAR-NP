from typing import Any, Optional

from neuralprocesses.data import AbstractGenerator


class NPDataLoader:
    def __init__(
        self, data_generator: AbstractGenerator, batch_size: Optional[int] = None
    ):
        """
        Initializes the data loader wrapper of the neural process data generation class.

        Args:
            data_generator: An instance of the custom data generation class.
            batch_size: The desired size of each batch to be returned by this loader.
        """
        self.data_generator = data_generator
        if batch_size is None:
            batch_size = data_generator.batch_size
        self.batch_size = batch_size
        self.data_buffer: list[Any] = []  # Buffer to accumulate data points

    def __iter__(self):
        """
        Returns the iterator for looping over the data.
        """
        return self

    def __next__(self):
        """
        Returns the next batch of the specified size.
        """
        while len(self.data_buffer) < self.batch_size:
            new_data = self.data_generator.generate_batch(self.batch_size)
            if new_data is None or len(new_data) == 0:
                # No more data to generate, handle partial or empty batch scenario
                if len(self.data_buffer) > 0:
                    # Return whatever is left in the buffer and clear it
                    batch = self.data_buffer
                    self.data_buffer = []
                    return batch
                else:
                    raise StopIteration  # End of data stream

            # Add new data to the buffer
            self.data_buffer.extend(new_data)

        # Gather enough data points to form a complete batch
        batch = self.data_buffer[: self.batch_size]
        self.data_buffer = self.data_buffer[self.batch_size :]
        return batch
