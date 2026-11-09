"""
Locust Performance & Load Testing Suite for Signature Detection & Verification API.
Simulates concurrent user workflows:
1. Health checks & model info polling.
2. Verification submission with document + reference image payloads.
3. Job status polling until completion.
"""

import os
from locust import HttpUser, task, between, tag


class SignatureVerificationUser(HttpUser):
    wait_time = between(0.5, 2.0)
    
    def on_start(self):
        """Prepare sample test document payload."""
        self.doc_path = "sample_test_run/sample_document.jpg"
        if not os.path.exists(self.doc_path):
            self.doc_path = "E:/sign_detect-verify/sample_test_run/sample_document.jpg"

    @tag("health")
    @task(3)
    def check_health(self):
        """Monitors health endpoint under load."""
        self.client.get("/health", name="/health")

    @tag("models")
    @task(2)
    def get_models_info(self):
        """Queries model version and architecture metadata."""
        self.client.get("/api/v1/models/info", name="/api/v1/models/info")

    @tag("metrics")
    @task(1)
    def scrape_metrics(self):
        """Simulates Prometheus scraper load."""
        self.client.get("/metrics", name="/metrics")

    @tag("verify")
    @task(5)
    def submit_and_poll_verification(self):
        """End-to-end user journey: Upload -> Poll -> Retrieve Results."""
        if not os.path.exists(self.doc_path):
            return

        with open(self.doc_path, "rb") as f_doc, open(self.doc_path, "rb") as f_ref:
            files = {
                "document": ("document.jpg", f_doc.read(), "image/jpeg"),
                "references": ("reference.jpg", f_ref.read(), "image/jpeg")
            }
            # Submit verification request
            with self.client.post("/api/v1/signatures/verify", files=files, catch_response=True, name="/api/v1/signatures/verify") as resp:
                if resp.status_code == 202:
                    job_id = resp.json().get("job_id")
                    resp.success()
                else:
                    resp.failure(f"Submission failed with status {resp.status_code}")
                    return

            # Poll for completion (up to 5 attempts)
            for _ in range(5):
                poll_resp = self.client.get(f"/api/v1/jobs/{job_id}", name="/api/v1/jobs/[id]")
                if poll_resp.status_code == 200:
                    status = poll_resp.json().get("status")
                    if status in ["COMPLETED", "FAILED"]:
                        break
