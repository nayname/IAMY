// step:3 file: Show last execution height for schedule daily_rewards
export const displayLastExecutionHeight = (height) => {
  if (height === undefined || height === null) {
    console.error('Height is not provided.');
    return;
  }
  console.log(`Last execution height: ${height}`);
  // You can additionally inject this into the DOM, e.g.,
  // document.getElementById('last-height').textContent = `Last execution height: ${height}`;
};


// step:1 file: Create a cron schedule named "daily_rewards" that distributes rewards every 7,200 blocks at END_BLOCKER
/* gatherScheduleInputs.js
 * Helper that can be wired to a form or wizard.
 */
export const gatherScheduleInputs = () => {
  // In a real app you would read these from form fields or a config file.
  const scheduleName = "daily_rewards";            // Unique schedule identifier
  const period = 7200;                              // Blocks between executions
  const executionStage = "EXECUTION_STAGE_END_BLOCKER"; // When to fire (Begin/End block)
  const targetContract = "neutron1contract...";     // Rewards contract address

  // CosmWasm execute payload that the cron job will run each period
  const rewardsMsg = {
    distribute: {}
  };

  // MsgExecuteContract that the Cron module will invoke
  const compiledExecuteMsg = {
    "@type": "/cosmwasm.wasm.v1.MsgExecuteContract",
    "sender": targetContract,         // will be overwritten by Cron when executed
    "contract": targetContract,
    "msg": Buffer.from(JSON.stringify(rewardsMsg)).toString("base64"),
    "funds": []
  };

  return {
    scheduleName,
    period,
    executionStage,
    authority: "neutron1mainDAOaddress...", // DAO (gov) address that controls Cron
    msgs: [compiledExecuteMsg]
  };
};


// step:1 file: Remove the cron schedule named "daily_rewards"
export const constructMsgRemoveSchedule = (authority, name = "daily_rewards") => {
  if (!authority) {
    throw new Error("Authority (DAO address) is required");
  }

  // EncodeObject compatible with CosmJS
  return {
    typeUrl: "/neutron.cron.MsgRemoveSchedule",
    value: {
      authority,
      name,
    },
  };
};
