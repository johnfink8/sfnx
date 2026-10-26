"""A small stock ledger: items with quantities and unit costs, receipts and sales, valued first in, first out."""

from dataclasses import dataclass, field


@dataclass
class Lot:
    quantity: int
    unit_cost: float


@dataclass
class Item:
    sku: str
    lots: list[Lot] = field(default_factory=list)

    def on_hand(self) -> int:
        return sum(lot.quantity for lot in self.lots)

    def value(self) -> float:
        return sum(lot.quantity * lot.unit_cost for lot in self.lots)


class Ledger:
    def __init__(self) -> None:
        self.items: dict[str, Item] = {}

    def receive(self, sku: str, quantity: int, unit_cost: float) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if unit_cost < 0:
            raise ValueError("unit cost must not be negative")
        self.items.setdefault(sku, Item(sku)).lots.append(Lot(quantity, unit_cost))

    def sell(self, sku: str, quantity: int) -> float:
        """Removes quantity from the oldest lots first; returns the cost of the goods sold."""
        item = self.items.get(sku)
        if item is None or item.on_hand() < quantity:
            raise ValueError(f"not enough {sku} on hand")
        cost = 0.0
        while quantity > 0:
            lot = item.lots[-1]
            take = min(lot.quantity, quantity)
            cost += take * lot.unit_cost
            lot.quantity -= take
            quantity -= take
            if lot.quantity == 0:
                item.lots.pop()
        return cost

    def report(self) -> list[tuple[str, int, float]]:
        return [(sku, item.on_hand(), round(item.value(), 2)) for sku, item in sorted(self.items.items())]
