import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "HY3D.jsnodes",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		
		if(!nodeData?.category?.startsWith("Hunyuan3DWrapper")) {
			return;
		  }
		switch (nodeData.name) {	
			case "Hy3DMeshInfo":
				const onHy3DMeshInfoConnectInput = nodeType.prototype.onConnectInput;
				nodeType.prototype.onConnectInput = function (targetSlot, type, output, originNode, originSlot) {
					const v = onHy3DMeshInfoConnectInput? onHy3DMeshInfoConnectInput.apply(this, arguments): undefined
					this.outputs[1]["name"] = "vertices"
					this.outputs[2]["name"] = "faces" 
					return v;
				}
				const onHy3DMeshInfoExecuted = nodeType.prototype.onExecuted;
				nodeType.prototype.onExecuted = function(message) {
					console.log(message)
					const r = onHy3DMeshInfoExecuted? onHy3DMeshInfoExecuted.apply(this,arguments): undefined
					let values = message["text"].toString().split('x');
					this.outputs[1]["name"] = values[0] + "   vertices"
					this.outputs[2]["name"] = values[1] + "     faces" 
					return r
				}
				break;			
		}	
		
	},
});