import cv2 as cv2
# sudo pip install untangle
import untangle

obj = untangle.parse('n00007846_6247.xml')

# Get Folder Name
print obj.annotation.folder.cdata

# Get Filename
print obj.annotation.filename.cdata

# Get Source Name
print obj.annotation.source.database.cdata

# Get Image Size (Width)
print obj.annotation.size.width.cdata
 
# Get Image Size (Height)
print obj.annotation.size.height.cdata

# Get Image Size (Depth)
print obj.annotation.size.depth.cdata

### Get Bounding Box of the Object
print obj.annotation.object.bndbox.xmin.cdata

### Get Bounding Box of the Object
print obj.annotation.object.bndbox.ymin.cdata

### Get Bounding Box of the Object
print obj.annotation.object.bndbox.xmax.cdata

### Get Bounding Box of the Object
print obj.annotation.object.bndbox.ymax.cdata

'''
<annotation>
	<folder>n00007846</folder>
	<filename>n00007846_6247</filename>
	<source>
		<database>ImageNet database</database>
	</source>
	<size>
		<width>500</width>
		<height>333</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>n00007846</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>161</xmin>
			<ymin>52</ymin>
			<xmax>285</xmax>
			<ymax>247</ymax>
		</bndbox>
	</object>
</annotation>'''