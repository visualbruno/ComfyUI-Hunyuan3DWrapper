import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

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
			case "Hy3DUploadMesh":
				addUploadWidget(nodeType, nodeData, "mesh");
				break;
		}	
		
	},
});

//file upload code from VHS nodes: https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
async function uploadFile(file) {
    //TODO: Add uploaded file to cache with Cache.put()?
    try {
        // Wrap file in formdata so it includes filename
        const body = new FormData();
        const i = file.webkitRelativePath.lastIndexOf('/');
        const subfolder = file.webkitRelativePath.slice(0,i+1)
        const new_file = new File([file], file.name, {
            type: file.type,
            lastModified: file.lastModified,
        });
        body.append("image", new_file);
        if (i > 0) {
            body.append("subfolder", subfolder);
        }
        const resp = await api.fetchApi("/upload/image", {
            method: "POST",
            body,
        });

        if (resp.status === 200) {
            return resp
        } else {
            alert(resp.status + " - " + resp.statusText);
        }
    } catch (error) {
        alert(error);
    }
}

function addUploadWidget(nodeType, nodeData, widgetName) {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        const pathWidget = this.widgets.find((w) => w.name === widgetName);
        const fileInput = document.createElement("input");
        chainCallback(this, "onRemoved", () => {
            fileInput?.remove();
        });
    	
		Object.assign(fileInput, {
			type: "file",
			accept: ".obj,.glb,.gltf,.stl,.3mf,.ply,model/obj,model/gltf-binary,model/gltf+json,application/vnd.ms-pki.stl,application/x-stl,application/vnd.ms-package.3dmanufacturing-3dmodel+xml,application/x-ply,application/ply",
			style: "display: none",
			onchange: async () => {
				if (fileInput.files.length) {
					let resp = await uploadFile(fileInput.files[0])
					if (resp.status != 200) {
						//upload failed and file can not be added to options
						return;
					}
					const filename = (await resp.json()).name;
					pathWidget.options.values.push(filename);
					pathWidget.value = filename;
					if (pathWidget.callback) {
						pathWidget.callback(filename)
					}
				}
			},
		});
        console.log(this)
        document.body.append(fileInput);
        let uploadWidget = this.addWidget("button", "choose glb file to upload", "image", () => {
            //clear the active click event
            app.canvas.node_widget = null

            fileInput.click();
        });
        uploadWidget.options.serialize = false;
    });
}

function chainCallback(object, property, callback) {
    if (object == undefined) {
        //This should not happen.
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object && object[property]) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r
        };
    } else {
        object[property] = callback;
    }
}