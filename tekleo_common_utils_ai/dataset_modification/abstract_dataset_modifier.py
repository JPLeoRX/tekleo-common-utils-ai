from tekleo_common_message_protocol import OdSample


class AbstractDatasetModifier:
    def apply(self, sample: OdSample) -> OdSample:
        pass
