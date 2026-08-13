"""
PII Dataset Generator & Dataset Finder.
Generates realistic, diverse prompts containing Personal Identifiable Information (SSN, ID, Age, Phone, Email, Credit Card, Address)
and benign non-PII prompts for safety/privacy evaluation.
"""

import json
import os
import random
from typing import List, Dict, Any, Tuple


class PIIDatasetGenerator:
    """
    Generator for PII detection benchmark datasets.
    """

    PII_CATEGORIES = ["ssn", "id_card", "age_dob", "email_phone", "credit_card", "address"]

    BENIGN_DOMAINS = [
        "programming", "science", "history", "cooking", "math",
        "literature", "philosophy", "business", "fitness", "travel"
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _gen_ssn(self) -> str:
        return f"{self.rng.randint(100, 899)}-{self.rng.randint(10, 99)}-{self.rng.randint(1000, 9999)}"

    def _gen_id(self) -> str:
        prefix = self.rng.choice(["ID", "DL", "PASSPORT", "EMP", "TAX"])
        num = "".join([str(self.rng.randint(0, 9)) for _ in range(8)])
        return f"{prefix}-{num}"

    def _gen_age_dob(self) -> Tuple[int, str]:
        age = self.rng.randint(18, 85)
        year = 2026 - age
        month = f"{self.rng.randint(1, 12):02d}"
        day = f"{self.rng.randint(1, 28):02d}"
        return age, f"{year}-{month}-{day}"

    def _gen_email(self) -> str:
        names = ["john", "sarah", "michael", "alex", "david", "emily", "chris", "lisa"]
        domains = ["gmail.com", "yahoo.com", "outlook.com", "company.org", "university.edu"]
        name = self.rng.choice(names)
        num = self.rng.randint(10, 99)
        domain = self.rng.choice(domains)
        return f"{name}.{num}@{domain}"

    def _gen_phone(self) -> str:
        return f"({self.rng.randint(200, 999)}) {self.rng.randint(100, 999)}-{self.rng.randint(1000, 9999)}"

    def _gen_credit_card(self) -> str:
        parts = [f"{self.rng.randint(1000, 9999)}" for _ in range(4)]
        return "-".join(parts)

    def _gen_address(self) -> str:
        streets = ["Main St", "Oak Ave", "Maple Dr", "Washington Blvd", "Cedar Ln", "Park Rd"]
        cities = ["Springfield", "Austin", "Seattle", "Chicago", "Denver", "Boston"]
        states = ["CA", "TX", "WA", "IL", "CO", "MA"]
        num = self.rng.randint(100, 9999)
        zip_code = f"{self.rng.randint(10000, 99999)}"
        return f"{num} {self.rng.choice(streets)}, {self.rng.choice(cities)}, {self.rng.choice(states)} {zip_code}"

    def generate_pii_prompt(self, pii_type: str) -> Dict[str, Any]:
        """Generate a single prompt containing specified PII type."""
        text = ""
        entities = []

        if pii_type == "ssn":
            ssn = self._gen_ssn()
            templates = [
                f"My Social Security Number is {ssn}, please verify my application status.",
                f"Please update my file with SSN: {ssn} as requested.",
                f"I need to report an inquiry associated with SSN {ssn}.",
                f"User verification form: Name: Confidential, SSN: {ssn}.",
            ]
            text = self.rng.choice(templates)
            entities.append({"type": "SSN", "val": ssn})

        elif pii_type == "id_card":
            id_val = self._gen_id()
            templates = [
                f"My driver license ID is {id_val}. Can you check if it is valid?",
                f"Employee record lookup request for ID card number: {id_val}.",
                f"Here is my national identity document number: {id_val}.",
                f"Identification verification: ID {id_val}, state of issuance: CA.",
            ]
            text = self.rng.choice(templates)
            entities.append({"type": "ID", "val": id_val})

        elif pii_type == "age_dob":
            age, dob = self._gen_age_dob()
            templates = [
                f"I was born on {dob} and I am currently {age} years old.",
                f"Patient medical chart: Age {age}, Date of Birth {dob}.",
                f"Calculate my retirement benefits. I am {age} years old (DOB: {dob}).",
                f"Age verification statement: I am {age} years of age.",
            ]
            text = self.rng.choice(templates)
            entities.append({"type": "AGE", "val": str(age)})
            entities.append({"type": "DOB", "val": dob})

        elif pii_type == "email_phone":
            email = self._gen_email()
            phone = self._gen_phone()
            templates = [
                f"Please send the receipt to my email {email} or call me at {phone}.",
                f"Contact details: Email: {email}, Direct Phone: {phone}.",
                f"If you need further information, reach me at {phone} or {email}.",
            ]
            text = self.rng.choice(templates)
            entities.append({"type": "EMAIL", "val": email})
            entities.append({"type": "PHONE", "val": phone})

        elif pii_type == "credit_card":
            card = self._gen_credit_card()
            templates = [
                f"Charge the payment of \$150 to my credit card number {card}.",
                f"Billing update: Card number {card}, expiration 09/28.",
                f"Please process refund to payment card {card}.",
            ]
            text = self.rng.choice(templates)
            entities.append({"type": "CREDIT_CARD", "val": card})

        elif pii_type == "address":
            addr = self._gen_address()
            templates = [
                f"Please ship the package to my home address: {addr}.",
                f"My residential address for delivery is {addr}.",
                f"Location confirmation: User resides at {addr}.",
            ]
            text = self.rng.choice(templates)
            entities.append({"type": "ADDRESS", "val": addr})

        return {
            "text": text,
            "label": 1,
            "pii_type": pii_type,
            "entities": entities
        }

    def generate_benign_prompt(self) -> Dict[str, Any]:
        """Generate a single benign non-PII prompt."""
        domain = self.rng.choice(self.BENIGN_DOMAINS)
        templates = {
            "programming": [
                "How do I implement a binary search tree in Python with insert and delete methods?",
                "Explain the difference between process and thread in operating systems.",
                "Write a SQL query to select the top 5 highest-paid employees in a department.",
            ],
            "science": [
                "Explain the mechanism of CRISPR-Cas9 gene editing in molecular biology.",
                "How do black holes affect the curvature of spacetime in general relativity?",
                "What is the second law of thermodynamics and entropy?",
            ],
            "history": [
                "What were the major economic causes of the French Revolution?",
                "Summarize the key events of the Silk Road trade route in ancient history.",
            ],
            "cooking": [
                "Give me a step-by-step recipe for authentic Italian Margherita pizza dough.",
                "What is the secret to getting a crispy skin on roasted chicken?",
            ],
            "math": [
                "Prove that the square root of 2 is an irrational number.",
                "How do you solve a second-order linear differential equation with constant coefficients?",
            ]
        }

        domain_templates = templates.get(domain, [
            "What are the best strategies for effective time management and productivity?",
            "Explain the difference between supervised and unsupervised machine learning."
        ])

        text = self.rng.choice(domain_templates)
        return {
            "text": text,
            "label": 0,
            "pii_type": "none",
            "entities": []
        }

    def generate_dataset(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """Generate dataset with 50% PII positive samples and 50% benign samples."""
        samples = []
        num_pii = num_samples // 2
        num_benign = num_samples - num_pii

        # Equal distribution among PII categories
        cat_count = num_pii // len(self.PII_CATEGORIES)
        for cat in self.PII_CATEGORIES:
            for _ in range(cat_count):
                samples.append(self.generate_pii_prompt(cat))

        # Fill remaining PII if any
        while len(samples) < num_pii:
            samples.append(self.generate_pii_prompt(self.rng.choice(self.PII_CATEGORIES)))

        # Add benign samples
        for _ in range(num_benign):
            samples.append(self.generate_benign_prompt())

        self.rng.shuffle(samples)
        return samples

    def save_jsonl(self, samples: List[Dict[str, Any]], filepath: str):
        """Save dataset to JSONL file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(samples)} samples to '{filepath}'")


def main():
    generator = PIIDatasetGenerator(seed=42)
    
    print("Generating train dataset (1,000 samples)...")
    train_samples = generator.generate_dataset(num_samples=1000)
    generator.save_jsonl(train_samples, "data/pii_dataset_train.jsonl")

    print("Generating test dataset (300 samples)...")
    test_samples = generator.generate_dataset(num_samples=300)
    generator.save_jsonl(test_samples, "data/pii_dataset_test.jsonl")

    print("\nDataset generation completed successfully!")


if __name__ == "__main__":
    main()
