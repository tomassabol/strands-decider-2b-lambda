#!/usr/bin/env node
import { App } from "aws-cdk-lib";
import { DeciderFunctionStack, stackNameFor } from "../lib/decider-function-stack";

const app = new App();

// One stack per model, named after it, so several models can be deployed side by side in one
// account and region.
new DeciderFunctionStack(app, stackNameFor(app.node.tryGetContext("model")), {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});
