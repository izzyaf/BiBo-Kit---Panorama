package org.opencv.stitchsample;

//import java.text.SimpleDateFormat;
//import java.util.Date;
import java.util.List;
import java.util.ListIterator;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.stitchsample.R;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.hardware.Camera.Size;
import android.os.Bundle;
import android.os.Environment;
//import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SubMenu;
import android.view.SurfaceView;
import android.view.View;
//import android.view.View.OnClickListener;
import android.view.View.OnTouchListener;
import android.view.WindowManager;
//import android.widget.Button;
import android.widget.Toast;

@SuppressLint("ClickableViewAccessibility")
public class StitchingActivity extends Activity implements
		CvCameraViewListener2, OnTouchListener{
	private static final String TAG = "OCVSample::Activity";

	private Tutorial3View mOpenCvCameraView;
	private List<Size> mResolutionList;
	private MenuItem[] mEffectMenuItems;
	private SubMenu mColorEffectsMenu;
	private MenuItem[] mResolutionMenuItems;
	private SubMenu mResolutionMenu;
	private int image_count;
	
	public native void stitchimage(int img_count);

	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");
				System.loadLibrary("stitchimage");
				mOpenCvCameraView.enableView();
				mOpenCvCameraView.setOnTouchListener(StitchingActivity.this);
			}
				break;
			default: {
				super.onManagerConnected(status);
			}
				break;
			}
		}
	};

	public StitchingActivity() {
		Log.i(TAG, "Instantiated new " + this.getClass());
	}

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.tutorial3_surface_view);
		image_count = 0;

		mOpenCvCameraView = (Tutorial3View) findViewById(R.id.tutorial3_activity_java_surface_view);

		mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

		mOpenCvCameraView.setCvCameraViewListener(this);
	}

	@Override
	public void onPause() {
		super.onPause();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	@Override
	public void onResume() {
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this,
				mLoaderCallback);
	}

	public void onDestroy() {
		super.onDestroy();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
	}

	public void onCameraViewStopped() {
	}

	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
		return inputFrame.rgba();
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		List<String> effects = mOpenCvCameraView.getEffectList();

		if (effects == null) {
			Log.e(TAG, "Color effects are not supported by device!");
			return true;
		}

		mColorEffectsMenu = menu.addSubMenu("Color Effect");
		mEffectMenuItems = new MenuItem[effects.size()];

		int idx = 0;
		ListIterator<String> effectItr = effects.listIterator();
		while (effectItr.hasNext()) {
			String element = effectItr.next();
			mEffectMenuItems[idx] = mColorEffectsMenu.add(1, idx, Menu.NONE,
					element);
			idx++;
		}

		mResolutionMenu = menu.addSubMenu("Resolution");
		mResolutionList = mOpenCvCameraView.getResolutionList();
		mResolutionMenuItems = new MenuItem[mResolutionList.size()];

		ListIterator<Size> resolutionItr = mResolutionList.listIterator();
		idx = 0;
		while (resolutionItr.hasNext()) {
			Size element = resolutionItr.next();
			mResolutionMenuItems[idx] = mResolutionMenu.add(2, idx, Menu.NONE,
					Integer.valueOf(element.width).toString() + "x"
							+ Integer.valueOf(element.height).toString());
			idx++;
		}

		return true;
	}

	public boolean onOptionsItemSelected(MenuItem item) {
		Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
		if (item.getGroupId() == 1) {
			mOpenCvCameraView.setEffect((String) item.getTitle());
			Toast.makeText(this, mOpenCvCameraView.getEffect(),
					Toast.LENGTH_SHORT).show();
		} else if (item.getGroupId() == 2) {
			int id = item.getItemId();
			Size resolution = mResolutionList.get(id);
			mOpenCvCameraView.setResolution(resolution);
			resolution = mOpenCvCameraView.getResolution();
			String caption = Integer.valueOf(resolution.width).toString() + "x"
					+ Integer.valueOf(resolution.height).toString();
			Toast.makeText(this, caption, Toast.LENGTH_SHORT).show();
		}

		return true;
	}

	//@SuppressLint("SimpleDateFormat")
	@Override
	public boolean onTouch(View v, MotionEvent event) {
		Log.i(TAG, "onTouch event");
		// SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
		// String currentDateandTime = sdf.format(new Date());
		// String fileName =
		// Environment.getExternalStorageDirectory().getPath()+
		// "/sample_picture_" + currentDateandTime + ".jpg";
		//String fileName = "/storage/emulated/0/panoTmpImage/im" + (image_count) + ".jpg";
		String sdcard = Environment.getExternalStorageDirectory().getPath();
		String fileName = sdcard + "/DCIM/Camera/im" + (image_count) + ".jpg";
		mOpenCvCameraView.takePicture(fileName);
		image_count++;
		if (image_count == 10) {
			stitchimage(image_count);
			image_count = 0;
		}
		
		Toast.makeText(this, fileName + " saved", Toast.LENGTH_SHORT).show();
		return false;
	}

	
}
